from random import shuffle
from random import randint
import numpy as np
import pandas as pd
import time


class Asset:
    """

    """

    def __init__(self, length, total_periods):
        """

        :param length:
        """
        self.length = int(length / 3)
        self.total_periods = total_periods
        self.diff_weight_num = total_periods - self.length + 1
        self.weights = []
        for i in range(total_periods):
            self.weights.append([])
        self.agg_weights = []

    def __str__(self):
        """

        :return:
        """
        return "Asset length: " + str(self.length) + "\n" + str(self.weights) + "\n" + str(self.agg_weights)

    def __eq__(self, other):
        """

        :param other:
        :return:
        """
        return self.weights == other.weights

    def calc_agg_weights(self):
        """

        :return:
        """
        for item in self.weights:
            self.agg_weights.append(sum(item))
        return self.agg_weights


def gen_random(rand_num, sum):
    """

    :param rand_num: int
    :param total: int
    :return: list
    """
    if sum == 0:
        return [0] * rand_num
    points = []
    for i in range(rand_num - 1):
        points.append(randint(0, sum))
    # print(points)
    points.sort()
    points.append(sum)
    points.insert(0, 0)
    # print(points)
    result = []
    for i in range(len(points) - 1):
        result.append(points[i + 1] - points[i])
    shuffle(result)
    # print(result)
    return result


def calc_variance(w, c):
    """

    :param w: list
    :param c: array
    :return:
    """
    # convert list to array
    variance = np.dot(w.T, np.dot(c, w))
    return variance


def calc_multi_variance(w, c, period_num, asset_num):
    """

    :param w: array
    :param c: array
    :param period_num: int
    :param asset_num: int
    :return: float
    """
    sum = 0
    for period in range(period_num):
        # slice weights of a specific period
        current_w = w[:, period]
        # slice a asset_num X asset_num matrix from c
        current_c = c[:, period * asset_num: period * asset_num + asset_num]
        # print(current_w)
        # print(current_c)
        # print()
        sum += calc_variance(current_w, current_c)
    return sum


def calc_multi_return(w, r, period_num, asset_num):
    """

    :param w: array
    :param r: array
    :param period_num: int
    :param asset_num: int
    :return: num
    """
    result = 0
    for period in range(period_num):
        # slice weights of a specific period
        current_w = w[:, period]
        # slice return of a specific period
        current_r = r[:, period]
        result += sum(current_w * current_r)
    return result


def generate_chromosome(levels, total_periods, asset_length):
    """

    :param levels: int
    :param total_periods: int
    :param asset_length: list of int
    :return: array
    """
    asset_num = len(asset_length)
    # construct asset list to store all Asset objects
    assets = []
    for i in range(len(asset_length)):
        assets.append(Asset(asset_length[i], total_periods))
    # print(assets)
    # calculate weights of each product period by period
    for period in range(total_periods):
        # the first period
        if period == 0:
            batch = gen_random(asset_num, levels)
            # assign batch to 20 assets as w1
            # loop over assets
            for i in range(asset_num):
                current_asset = assets[i]
                # loop over weight list for necessary periods
                for j in range(period, min(total_periods, period + current_asset.length)):
                    current_asset.weights[j].append(batch[i])
        # print(assets[0])
        # all periods after the first period need to consider previous periods results
        else:
            # print("current period: " + str(period))
            # print(assets[3])
            # first select assets that need to add weights
            # also compute the current period sum of all existing weights for all assets
            needy_assets = []
            existing_weights = 0
            for asset in assets:
                # print("=====")
                # print(asset.length)
                if period < asset.diff_weight_num:
                    needy_assets.append(asset)
                # print(asset.weights[period])
                existing_weights += sum(asset.weights[period])
            # print("existing weights: " + str(existing_weights))
            remaining_weights = max(0, levels - existing_weights)
            # print("remaining weights: " + str(remaining_weights))
            # print("Is assets[0] in needy: " + str(assets[3] in needy_assets))
            # generate random weights for needy assets
            needy_batch = gen_random(len(needy_assets), remaining_weights)
            # print('needy_batch' + str(needy_batch))
            # assign needy_batch to needy assets
            # loop over needy assets
            for i in range(len(needy_assets)):
                current_asset = needy_assets[i]
                # loop over weight list for necessary periods
                for j in range(period, min(total_periods, period + current_asset.length)):
                    current_asset.weights[j].append(needy_batch[i])

                    # print()

    # calculate agg_weights for every asset and collect all weights into port_weights
    port_weights = []
    for asset in assets:
        port_weights.append(asset.calc_agg_weights())

    # convert port_weights into array
    port_weights = np.array(port_weights)
    return port_weights


def calc_pop_fitness(population, r, c, total_periods, asset_num):
    """

    :param population: list of array
    :param port_weights: array
    :param r: array
    :param c: array
    :param total_periods: int
    :param asset_num: int
    :return: list of num
    """
    fitness = []
    for port_weights in population:
        port_variance = calc_multi_variance(port_weights, c, total_periods, asset_num)
        port_return = calc_multi_return(port_weights, r, total_periods, asset_num)
        fitness.append(port_return - port_variance)
    return fitness


def select_mating_pool(new_population, fitness, num_parents_mating):
    # deep copy fitness list
    fitness_copy = []
    for item in fitness:
        fitness_copy.append(float(item))
    fitness_copy.sort()
    mating_pool = []
    for i in range(num_parents_mating):
        # get the fitness from back
        mating_pool.append(new_population[fitness.index(fitness_copy[-i - 1])])
    return mating_pool


def crossover(parents, offspring_size):
    """

    :param parents: array
    :param offspring_size: int
    :return: array
    """
    shuffle(parents)
    offsprings = []
    crossover_point = 6
    # find 12 period product index
    index_12 = []
    example = parents[0]
    for row in range(len(example)):
        first_element = example[row, 0]
        flag = True
        for col in range(len(example[0])):
            if example[row, col] != first_element:
                flag = False
                break
        if flag == True:
            index_12.append(row)
    for i in range(int(offspring_size / 2)):
        offspring1 = []
        offspring2 = []
        for row in range(len(parents[i])):
            if row in index_12:
                offspring1.append(list(parents[i][row]))
                offspring2.append(list(parents[i + 1][row]))
            else:
                cross1 = list(parents[i][row, :crossover_point]) + list(parents[i + 1][row, crossover_point:])
                offspring1.append(cross1)
                cross2 = list(parents[i + 1][row, :crossover_point]) + list(parents[i][row, crossover_point:])
                offspring2.append(cross2)
                # # first offspring 1, 4
                # offsprings.append(np.concatenate((parents[i][:,:crossover_point],parents[i+1][:,crossover_point:]),1))
                # # second offspring 3, 2
                # offsprings.append(np.concatenate((parents[i+1][:,:crossover_point], parents[i][:,crossover_point:]), 1))
        offspring1 = np.array(offspring1)
        offspring2 = np.array(offspring2)
        offsprings.append(offspring1)
        offsprings.append(offspring2)
    # print("-----")
    # print(offsprings)
    return offsprings


def find_index_12(parents):
    """

    :param parents: list of arrays
    :return: list of int
    """
    # find 12 period product index
    index_12 = []
    example = parents[0]
    for row in range(len(example)):
        first_element = example[row, 0]
        flag = True
        for col in range(len(example[0])):
            if example[row, col] != first_element:
                flag = False
                break
        if flag == True:
            index_12.append(row)
    return index_12


def mutation(offspring_crossover, index_12, levels):
    """

    :param offspring_crossover:
    :return:
    """
    # .shape return a tupple of and row_num and col_num
    col_num = offspring_crossover[0].shape[1]
    num_12_product = len(index_12)
    # mutate half of the 12 products
    mut_12_num = num_12_product // 2
    mut_12_index = []
    for i in range(mut_12_num):
        rand_index = randint(0, len(index_12) - 1)
        mut_12_index.append(index_12[rand_index])
    possible_mut = [-2, -1, 0, 0, 1, 2]
    for offspring in offspring_crossover:
        for row in mut_12_index:
            this_mut = possible_mut[randint(0, len(possible_mut) - 1)]
            if int(offspring[row, 0]) + this_mut > 0:
                offspring[row, :] = offspring[row, :] + this_mut

    for offspring in offspring_crossover:
        for col in range(col_num):
            # calculate current sum of the this column
            sum = 0
            for row in range(len(offspring)):
                sum += int(offspring[row, col])
            # get vacant levels
            remainder = levels - sum

            # construct possible values for mutation
            if remainder == 0:
                mut_list = [0, 0, 0, 0, 0, -1, -2, -3, -4, -5]
            elif 0 < remainder <= 5:
                mut_list = [0, 0, 0, -1, -2, -3, -4, -5]
                for i in range(remainder):
                    mut_list.append(i)
            else:
                mut_list = [-5, -4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4]
            remaining = remainder
            # loop over every element in this column
            mutation_number = 10
            for i in range(mutation_number):
                row = randint(0, len(offspring) - 1)
                if row in index_12:
                    continue
                else:
                    mut_num = mut_list[randint(0, len(mut_list) - 1)]
                    current_num = int(offspring[row, col])
                    # check if mutation can happen
                    new_remaining = remaining - mut_num
                    if new_remaining >= 0:
                        new_num = max(0, current_num + mut_num)
                        offspring[row, col] = new_num
                        remaining = new_remaining

    for offspring in offspring_crossover:
        for row in range(len(offspring)):
            for col in range(col_num):
                current = int(offspring[row, col])
                if current < 0:
                    offspring[row, col] = 0
    return offspring_crossover


if __name__ == "__main__":
    # load covariance matrix
    df = pd.read_csv('CovarianceMatrix_50X20X240.csv', header=None)
    total_c = np.array(df)

    # load return data
    df = pd.read_csv('Returns_50X20X12.csv', header=None)
    total_r = np.array(df)

    # load length data
    df = pd.read_csv('LoanDuration_20X50.csv', header=None)
    total_l = np.array(df)

    # define population related parameters
    levels = 127
    total_periods = 12

    solution = []
    result = []

    for i in range(15):
        print("=================================================")
        print("Asset Number: " + str(i))
        c = total_c[20*i:20*(i+1),:]
        r = total_r[20*i:20*(i+1),:]
        asset_length = list(total_l[:,i])
        asset_num = len(asset_length)

        # print(c)
        # print()
        # print(r)
        # print()
        # print(asset_length)


        # generate_chromosome(levels, total_periods, [18,12,36])

        # define genetic algorithm parameters 8192  4096
        sol_per_pop = 1024
        num_parents_mating = 512

        # generate new population
        new_population = []
        for i in range(sol_per_pop):
            new_population.append(generate_chromosome(levels, total_periods, asset_length))
        # print(new_population)
        #

        start = time.time()

        num_generations = 200
        for generation in range(num_generations):
            print("Generation : ", generation)
            # Measuring the fitness of each chromosome in the population.
            fitness = calc_pop_fitness(new_population, r, c, total_periods, asset_num)
            # Selecting the best parents in the population for mating.
            parents = select_mating_pool(new_population, fitness, num_parents_mating)
            # Generating next generation using crossover.
            # print(parents)
            # Creating the new population based on the parents and offspring.
            # offspring size can not be more than twice as many as parents
            offspring_crossover = crossover(parents, sol_per_pop - num_parents_mating)

            # find index_12
            index_12 = find_index_12(parents)
            # mutation
            offspring_mutation = mutation(offspring_crossover, index_12, levels)
            # create new population
            new_population = parents + offspring_mutation

            # The best result in the current iteration.
            print("Best result: " + str(max(fitness)))

        result_fitness = calc_pop_fitness(new_population, r, c, total_periods, asset_num)
        parents = select_mating_pool(new_population, result_fitness, num_parents_mating)

        current_rar = max(result_fitness)

        end = time.time()

        current_solution = new_population[result_fitness.index(current_rar)]

        current_return = calc_multi_return(current_solution, r, total_periods, asset_num)

        current_variance = calc_multi_variance(current_solution, c, total_periods, asset_num)

        current_sharpe = current_return/current_variance

        solution += list(current_solution)
        result.append([current_return, current_variance, current_rar, current_sharpe])


        print("Final result: " + str(current_rar)[1:])
        print("Final solution: " + "\n" + str(new_population[result_fitness.index(current_rar)]))
        runtime = end - start
        print("Run time: ")
        print(runtime)
        # Getting the best solution after iterating finishing all generations.

    solution_csv = pd.DataFrame(solution)

    solution_csv.to_csv("solution.csv", sep =',')

    result_csv = pd.DataFrame(result)

    result_csv.to_csv("result.csv", sep =',')







