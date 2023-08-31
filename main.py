import numpy as np 
import pandas as pd
import random
import copy
import os
import sys
from ECVRP import TwoECVrp, scatter_search_vns, ReOptimization, time_penalty
import xlsxwriter
import networkx as nx 
import matplotlib.pyplot as plt
from demand.knn import KNN
import csv


def output_csv_plot(dat_file,max_iterations, neighborhood_size):
    #Delete All old file from CSV Folder
    for file in os.listdir('CSV'):
        os.remove(os.path.join('CSV', file))
    path = os.path.join('excel_params', dat_file)
    dat_file1 = dat_file[0:-4]
    twoecvrp = TwoECVrp(path)
    num_solutions = twoecvrp.n_satellite + 1
    best_sols = [scatter_search_vns(twoecvrp, i + 1, num_solutions, neighborhood_size) for i in range(max_iterations)]
    best_cost = np.array([twoecvrp.calculate_cost(best_sol) for best_sol in best_sols])
    print('failed route')
    print(best_sols[-1])
    twoecvrp.add_labels(best_sols[-1])
    # Plotting------
    steps = np.arange(1, max_iterations + 1)
    
    plt.plot(steps, best_cost)
    plt.xlabel('iterations')
    plt.ylabel('Fitness')
    plt.title('Optimization Process')
    folder = 'Visualization&Steps'
    title = 'failed_fitness_iteration' + '.png'
    plt.savefig(os.path.join(folder, title))
    #plt.show()
    #---------------
    cus_index_list = list(np.arange(twoecvrp.n_customers))
    # Save the Expected Demand --------------
    if os.path.isfile('demand/expected_demand_history.csv') == False:
        with open('demand/expected_demand_history.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['D'+str(i) for i in cus_index_list])
    with open('demand/expected_demand_history.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(twoecvrp.expected_demand))
        
    # Simulate and Save the real demand ------
    real_demand = np.zeros(twoecvrp.n_customers)
    for i in range(twoecvrp.n_customers):
        rand_demand = np.random.normal(twoecvrp.demand[i, 1], (twoecvrp.demand[i, 2] - twoecvrp.demand[i, 0])/6)
        real_demand[i] = np.int64(min(twoecvrp.demand[i, 2], max(twoecvrp.demand[i, 0], rand_demand)))
    
    if os.path.isfile('demand/demand_history.csv') == False:
        with open('demand/demand_history.csv', 'w', newline='') as file:
            writer = csv.writer(file)
    with open('demand/demand_history.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(real_demand))
    #------------------------------------------
    
    # Re-Optimization ----------
    failed_time, stopped_solution, failed_cus_list, rest_of_goods, restart_time = twoecvrp.catch_failed_statement(solution=best_sols[-1], real_demand=real_demand)
    print(stopped_solution)
    print(failed_cus_list)
    if len(failed_cus_list) == 0:
        print('Complete Time is ' + str(failed_time))
        
        print(twoecvrp.expected_demand)
        print(real_demand)
        print('All demands are satisfied by the first transportation')
    else:
        re_twoecvrptw = ReOptimization(path, restart_time, failed_cus_list, current_sat_amount=rest_of_goods)
        print(re_twoecvrptw.sat_to_cus_distances)
        re_best_sols = [scatter_search_vns(re_twoecvrptw, i + 1, num_solutions, neighborhood_size) for i in range(max_iterations)]
        re_best_cost = np.array([re_twoecvrptw.calculate_cost(re_best_sol) for re_best_sol in re_best_sols])



        ## Reoptimization Plotting --------
        steps1 = steps
        plt.plot(steps1, re_best_cost)
        plt.xlabel('iterations')
        plt.ylabel('Fitness')
        plt.title('Re-Optimization Process')
        folder = 'Visualization&Steps'
        title = 'reoptimized_fitness_iteration' + '.png'
        plt.savefig(os.path.join(folder, title))
    
    #---------------------------
        print('Note: -1 is the depot, other indice are satellite')
   
        # Visualize the Route
        twoecvrp.plot_graph(best_sols[-1])
        re_twoecvrptw.plot_graph(re_best_sols[-1])

    # Summarization ------------
    if os.path.isfile('CSV/summarization.xlsx'):
        os.remove('CSV/summarization.xlsx')
    workbook = xlsxwriter.Workbook('CSV/summarization.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, '目标函数')
    worksheet.write(1, 0, '一级网络车辆固定成本')
    worksheet.write(2, 0, '一级网络车辆运输成本')
    worksheet.write(3, 0, '二级网络车辆固定成本')
    worksheet.write(4, 0, '二级网络车辆运输成本')
    worksheet.write(5, 0, '时间惩罚')
    worksheet.write(6, 0, '运行时间')
    worksheet.write(0, 1, best_cost[-1])
    worksheet.write(1, 1, twoecvrp.fixed_cost_1)
    worksheet.write(2, 1, twoecvrp.unit_cost_1)
    worksheet.write(3, 1, twoecvrp.fixed_cost_2)
    worksheet.write(4, 1, twoecvrp.unit_cost_2)
    worksheet.write(5, 1, twoecvrp.sum_penalty(best_sols[-1]))
    worksheet.write(6, 1, max(twoecvrp.arrive_time(best_sols[-1])))
    worksheet.write(7, 0, 'Planed Route')
    worksheet.write(8, 0, 'Depot to Satellite Route')
    worksheet.write(9, 0, 'Urbans ID')
    worksheet.write(9, 1, 'route')
    worksheet.write(9, 2, 'distance')
    worksheet.write(9, 3, 'Amount of Goods')
    worksheet.write(9, 4, 'Arrive & Leave Time')
    row = 10
    route1 = best_sols[-1]['depot_to_sat']
    
    route1_label = twoecvrp.add_labels(best_sols[-1])
    capacity = twoecvrp.get_urban_info(best_sols[-1])
    for i in range(twoecvrp.n_satellite):
        if len(route1_label['depot_to_sat'][i]) > 0:
            
            for j in range(len(route1_label['depot_to_sat'][i])):
                worksheet.write(row, 0, 'Urban' + str(row-10))
                worksheet.write(row, 1, route1_label['depot_to_sat'][i][j])
                worksheet.write(row, 3, capacity[i] / len(route1_label['depot_to_sat'][i]))
                worksheet.write(row, 2, twoecvrp.depot_to_sat_distances[i])
                worksheet.write(row, 4, str((twoecvrp.depot_to_sat_distances[i], twoecvrp.depot_to_sat_distances[i] + twoecvrp.st_satellite[i])))
                row += 1
                
    worksheet.write(row, 0, 'Satellite to Customer Route')
    row += 1
    worksheet.write(row, 0, 'Vehicles ID')
    worksheet.write(row, 1, 'route')
    worksheet.write(row, 2, 'distance')
    worksheet.write(row, 3, 'Amount of Goods')
    worksheet.write(row, 4, 'Arrive & Leave Time')
    row += 1
    idx = 0
    route2 = best_sols[-1]['sat_to_cus']
    route2_label = route1_label['sat_to_cus']
    arrive_time = twoecvrp.arrive_time(best_sols[-1])
    for i in range(twoecvrp.n_satellite):
        if len(route2_label[i]) > 0:
            for j in range(len(route2_label[i])):
                worksheet.write(row, 0, 'Vehicle' + str(idx))
                worksheet.write(row, 1, route2_label[i][j])
                worksheet.write(row, 2, twoecvrp.get_vehicle_dis(i, route2[i][j])[0])
                worksheet.write(row, 3, twoecvrp.get_vehicle_dis(i, route2[i][j])[1])
                worksheet.write(row, 4, str([(arrive_time[cus], arrive_time[cus] + twoecvrp.st_customer[cus]) for cus in route2[i][j]]))
                row += 1
                idx += 1
                
    ## Re-Optimization
    worksheet.write(row, 0, 'Re-Optimization')
    row += 1
    worksheet.write(row, 0, 'Stopped_Time')
    worksheet.write(row, 1, failed_time)
    row += 1
    worksheet.write(row, 0, 'failed customer Index')
    worksheet.write(row, 1, str(failed_cus_list))
    row += 1
    worksheet.write(row, 0, 'Remained Amount of Satellite')
    worksheet.write(row, 1, str(rest_of_goods))
    row += 1
    worksheet.write(row, 0, 'Total Time')
    worksheet.write(row, 1, np.max(re_twoecvrptw.arrive_time(re_best_sols[-1])))
    row += 1
    worksheet.write(row, 0, 'Penalty')
    worksheet.write(row, 1, re_twoecvrptw.sum_penalty(re_best_sols[-1]))
    row += 1
    worksheet.write(row, 0, "Cost of Re-Optimization")
    worksheet.write(row, 1, re_best_cost[-1])
    row += 1
    worksheet.write(row, 0, 'Re-Route of Urbans')
    row += 1
    worksheet.write(row, 0, 'Urbans ID')
    worksheet.write(row, 1, 'route')
    worksheet.write(row, 2, 'distance')
    worksheet.write(row, 3, 'Amount of Goods')
    worksheet.write(row, 4, 'Arrive & Leave Time')
    row += 1
    re_idx1 = 0
    re_route1 = re_best_sols[-1]['depot_to_sat']
    re_route1_label = re_twoecvrptw.add_labels(re_best_sols[-1])
    re_capacity = re_twoecvrptw.get_urban_info(re_best_sols[-1])
    for i in range(re_twoecvrptw.n_satellite):
        if len(re_route1_label['depot_to_sat'][i]) > 0:
            
            for j in range(len(re_route1_label['depot_to_sat'][i])):
                worksheet.write(row, 0, 're-Urban' + str(re_idx1))
                worksheet.write(row, 1, re_route1_label['depot_to_sat'][i][j])
                worksheet.write(row, 3, re_capacity[i] / len(re_route1_label['depot_to_sat'][i]))
                worksheet.write(row, 2, re_twoecvrptw.depot_to_sat_distances[i])
                worksheet.write(row, 4, str((re_twoecvrptw.depot_to_sat_distances[i] + restart_time[i] + failed_time, re_twoecvrptw.depot_to_sat_distances[i] + restart_time[i] + failed_time + re_twoecvrptw.st_satellite[i])))
                row += 1
                re_idx1 += 1
    
    worksheet.write(row, 0, 'Re-Route of Vehicles')
    row += 1
    worksheet.write(row, 0, 'Vehicles ID')
    worksheet.write(row, 1, 'route')
    worksheet.write(row, 2, 'distance')
    worksheet.write(row, 3, 'Amount of Goods')
    worksheet.write(row, 4, 'Arrive & Leave Time')
    row += 1
    re_idx2 = 0
    re_route2 = re_best_sols[-1]['sat_to_cus']
    re_route2_label = re_route1_label['sat_to_cus']
    re_arrive_time = re_twoecvrptw.arrive_time(re_best_sols[-1])
    for i in range(re_twoecvrptw.n_satellite):
        if len(re_route2_label[i]) > 0:
            for j in range(len(re_route2_label[i])):
                worksheet.write(row, 0, 'Re-Vehicle' + str(re_idx2))
                worksheet.write(row, 1, re_route2_label[i][j])
                worksheet.write(row, 2, re_twoecvrptw.get_vehicle_dis(i, re_route2[i][j])[0])
                worksheet.write(row, 3, re_twoecvrptw.get_vehicle_dis(i, re_route2[i][j])[1])
                worksheet.write(row, 4, str([(re_arrive_time[cus], re_arrive_time[cus] + re_twoecvrptw.st_customer[cus]) for cus in re_route2[i][j]]))
                row += 1
                re_idx2 += 1
    
    worksheet.write(row, 0, 'Total Cost')
    worksheet.write(row, 1, best_cost[-1] + re_best_cost[-1])
    row += 1
    worksheet.write(row, 0, 'Total Penalty')
    worksheet.write(row, 1, twoecvrp.sum_penalty(best_sols[-1]) + re_twoecvrptw.sum_penalty(re_best_sols[-1]))
    workbook.close()
    
Data_Folder = 'excel_params'
Plot_Folder = 'Visualization'
for file in os.listdir(Data_Folder):
    #path = os.path.join(Data_Folder, file) 
    max_iteration = 3
    neighborhood_size = 2
    output_csv_plot(file, max_iteration, neighborhood_size)
    
