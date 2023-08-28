import numpy as np 
import pandas as pd 
import xlsxwriter
def outPut(model,run_time,basename):
    work=xlsxwriter.Workbook(basename+'-result-0.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0, 0, '目标函数')
    worksheet.write(1, 0, '一级网络车辆固定成本')
    worksheet.write(2, 0, '一级网络车辆运输成本')
    worksheet.write(3, 0, '二级网络车辆固定成本')
    worksheet.write(4, 0, '一级网络车辆运输成本')
    worksheet.write(5, 0, '时间惩罚')
    worksheet.write(6, 0, '运行时间')
    worksheet.write(0, 1, model.best_sol.obj)
    worksheet.write(1, 1, model.best_sol.truck_fix_cost)
    worksheet.write(2, 1, model.best_sol.truck_var_cost)
    worksheet.write(3, 1, model.best_sol.car_fix_cost)
    worksheet.write(4, 1, model.best_sol.car_var_cost)
    worksheet.write(5, 1, model.best_sol.time_penalty)
    worksheet.write(6, 1, run_time)
    worksheet.write(7, 0, ' ')
    worksheet.write(8, 0, 'id')
    worksheet.write(8, 1, '一级网络路径')
    worksheet.write(8, 2, '路径距离')
    worksheet.write(8, 3, '配送货物量')
    worksheet.write(8, 4, '时刻')
    row = 9
    for id,route in enumerate(model.best_sol.truck_route_list):
        route_str=[str(i)for i in route]
        route_loaded_str=[str(i) for i in model.best_sol.truck_route_loaded_list[id]]
        route_time_str = [str(i) for i in model.best_sol.truck_route_timetable[id]]
        worksheet.write(row, 0, f'v{id}')
        worksheet.write(row, 1, '-'.join(route_str))
        worksheet.write(row, 2, model.best_sol.truck_route_distance[id])
        worksheet.write(row, 3, '-'.join(route_loaded_str))
        worksheet.write(row, 4, '-'.join(route_time_str))
        row += 1
    worksheet.write(row, 0, ' ')
    row += 1
    worksheet.write(row, 0, 'id')
    worksheet.write(row, 1, '二级网络路径')
    worksheet.write(row, 2, '路径距离')
    worksheet.write(row, 3, '配送货物量')
    worksheet.write(row, 4, '时刻')
    row += 1
    for id, route in enumerate(model.best_sol.car_route_list):
        route_str = [str(i) for i in route]
        route_time_str = [str(i) for i in model.best_sol.car_route_timetable[id]]
        worksheet.write(row, 0, f'v{id}')
        worksheet.write(row, 1, '-'.join(route_str))
        worksheet.write(row, 2, model.best_sol.car_route_distance[id])
        worksheet.write(row, 3, model.best_sol.car_route_loaded_list[id])
        worksheet.write(row, 4, '-'.join(route_time_str))
        row += 1
    worksheet.write(row, 0, '违反约束的节点')
    node_id = [str(i) for i in model.best_sol.abnormal_customers]
    worksheet.write(row, 1, '-'.join(node_id))
    worksheet = work.add_worksheet('算子信息')
    worksheet.write(0, 0, 'random destory weight')
    worksheet.write(1, 0, 'random destory select')
    worksheet.write(2, 0, 'random destory score')
    worksheet.write(0, 1, model.d_weight[0])
    worksheet.write(1, 1, model.d_history_select[0])
    worksheet.write(2, 1, model.d_history_score[0])

    worksheet.write(3, 0, 'worse destory weight')
    worksheet.write(4, 0, 'worse destory select')
    worksheet.write(5, 0, 'worse destory score')
    worksheet.write(3, 1, model.d_weight[1])
    worksheet.write(4, 1, model.d_history_select[1])
    worksheet.write(5, 1, model.d_history_score[1])

    worksheet.write(6, 0, 'random repair weight')
    worksheet.write(7, 0, 'random repair select')
    worksheet.write(8, 0, 'random repair score')
    worksheet.write(6, 1, model.r_weight[0])
    worksheet.write(7, 1, model.r_history_select[0])
    worksheet.write(8, 1, model.r_history_score[0])

    worksheet.write(9,  0, 'greedy repair weight')
    worksheet.write(10, 0, 'greedy repair select')
    worksheet.write(11, 0, 'greedy repair score')
    worksheet.write(9,  1, model.r_weight[1])
    worksheet.write(10, 1, model.r_history_select[1])
    worksheet.write(11, 1, model.r_history_score[1])

    worksheet.write(12, 0, 'regret repair weight')
    worksheet.write(13, 0, 'regret repair select')
    worksheet.write(14, 0, 'regret repair score')
    worksheet.write(12, 1, model.r_weight[1])
    worksheet.write(13, 1, model.r_history_select[1])
    worksheet.write(14, 1, model.r_history_score[1])

    work.close()