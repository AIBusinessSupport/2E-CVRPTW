# Import necessary libraries
import numpy as np
import random
import copy
import scipy
import pandas as pd
import sklearn
import networkx as nx 
import matplotlib.pyplot as plt
from demand.knn import KNN
## Define Path for Code Testing
path = 'excel_params/Ca1_2_3_15.csv'   
# Define the 2E-VRP class

def time_penalty(t, time_window):
    if t < time_window[0] or t > time_window[3]:
        return 0
    if t >= time_window[0] and t < time_window[1]:
        return (t - time_window[0]) / (time_window[1] - time_window[0])
    if t >= time_window[1] and t <= time_window[2]:
        return 1
    if t > time_window[2] and t <= time_window[3]:
        return (time_window[3] - t) / (time_window[3] - time_window[2])
        
class TwoECVrp:
    
    def __init__(self, path):
        # Set the number of customers and depots
        self.path = path
        params_dict = self.upload_params_from_excel()
        self.n_customers = len(params_dict['coor_custo'])
        self.n_satellite = len(params_dict['coor_satellite'])
        
        # Generate random demand for each customer
        self.demand = params_dict['fuzzy_demand']
        self.expected_demand = KNN(self.demand, params_dict['coor_custo'])
        self.real_demand = self.demand[:, 1]
        #self.expected_demand = self.demand[:,1]
        self.points = [tuple(params_dict['coor_depot'][0])]
        self.labels = {'D'}
        # Set the time windows of each customer
        self.time_window = params_dict['time_window']
        
        # Set the Service Time of Satellite and Customers
        self.st_satellite = params_dict['st_satellite']
        self.st_customer = params_dict['st_customer']
        
        # Generate random distances between all nodes
        self.depot_to_sat_distances = np.zeros(self.n_satellite) # Distance between depot and satellite
        self.sat_to_cus_distances = np.zeros((self.n_satellite, self.n_customers)) # Distance between satellite and customers
        self.sat_to_sat_distances = np.zeros((self.n_satellite, self.n_satellite))
        
        # Set the commissions of each satellite
        self.hs = np.zeros(self.n_satellite)
        for i in range(self.n_satellite):
            #self.sat_cap[i] = params_dict['coor_cap_cost_satellite'][i][3]
            #self.hs[i] = params_dict['coor_cap_cost_satellite'][i][4]
            self.labels[i+1] = 'Sat' + str(i)
            self.points.append(tuple(params_dict['coor_satellite'][i]))
            coor_sati = np.array(params_dict['coor_satellite'][i])
            coor_depot = np.array(params_dict['coor_depot'][0])
            self.depot_to_sat_distances[i] = np.sqrt(sum((coor_sati - coor_depot) ** 2))
            
            for j in range(self.n_customers):
                coor_cus = np.array(params_dict['coor_custo'][j])
                self.sat_to_cus_distances[i, j] = np.sqrt(sum((coor_sati - coor_cus) ** 2))
            
            for s in range(self.n_satellite):
                coor_sats = np.array(params_dict['coor_satellite'][s])
                self.sat_to_sat_distances[i, s] = np.sqrt(sum((coor_sats - coor_sati) ** 2))    
                
        ## Define the distance between cus to cus
        self.cus_to_cus_distances = np.zeros((self.n_customers, self.n_customers))
        for i in range(self.n_customers):
            self.labels[i + 1 + self.n_satellite] = 'Cus' + str(i)
            self.points.append(tuple(params_dict['coor_custo'][i]))
            for j in range(self.n_customers):
                coori = np.array(params_dict['coor_custo'][i])
                coorj = np.array(params_dict['coor_custo'][j])
                self.cus_to_cus_distances[i, j] = np.sqrt(sum((coori - coorj)**2))

        # Set the capacity of satellite (But Optional)
        self.sat_cap = params_dict['max_cap_satellite']
        
        # Set the capacity of the vehicles
        self.vehicle1_cap = params_dict['Q1']
        self.vehicle2_cap = params_dict['Q2'] 
        
        # Set the number of vehicles available at each depot
        #self.vehicle1_num = params_dict['truck_depot'][0]
        #self.vehicle2_num = params_dict['truck_satellite'][0]
        
        # Set the limit pass of each satellite
        #self.sat_pass = params_dict['max_pass_satellite'][0]
        
        # Set the cost the unit
        self.unit_cost_1 = params_dict['cost_ds_FE']
        self.unit_cost_2 = params_dict['cost_ds_SE']
        self.fixed_cost_1 = params_dict['fix_cost_FE']
        self.fixed_cost_2 = params_dict['fix_cost_SE']
        
        # Initialize the solution with a random assignment of customers to depots
        sat_solution = self.generate_init_sat_solutions()[0]
        self.solution = self.generate_depot_solutions(sat_solution)
        
        # Calculate the cost of the initial solution
        self.cost = self.calculate_cost(self.solution)
    
    def upload_params_from_excel(self):
        #Load the Data from CSV file. 
        df = pd.read_csv(self.path)
        col_list = df.columns
        params_dict = {
            'Q1': 200,
            'cost_ds_FE': 1,
            'fix_cost_FE': 50,
            'st_satellite': [],
            'Q2': 50,
            'cost_ds_SE': 1,
            'fix_cost_SE': 25,
            'st_customer': [],
            'coor_depot': [],
            'max_cap_satellite': [],
            'coor_satellite': [],
            'coor_custo': [],
            'fuzzy_demand': [],
            'time_window': []
            
        }
        '''
        Assume that
                Urban vehicle capacity=200 
                Urban vehicle cost=50
                City freighter capacity=50 
                City freighter cost=25
        '''
        
        ## Get the Parameters from CSV file
        depot = df[df['mark'] == 0]
        satellite = df[df['mark'] == 1]
        customer = df[df['mark'] == 2]
        
        ## Coordincate of Depot
        coor_depot = depot[col_list[1:3]].values
        params_dict['coor_depot'] = coor_depot.tolist()
        
        ## Coordinate of Satellite
        coor_sat = satellite[col_list[1:3]].values
        st_sat = satellite[col_list[-2]].values
        params_dict['coor_satellite'] = coor_sat.tolist()
        params_dict['st_satellite'] = st_sat.tolist()
            
        ## Coordinate of Customers
        st_cus = customer[col_list[-2]].values
        coor_cus = customer[col_list[1:3]].values
        params_dict['coor_custo'] = coor_cus.tolist()
        params_dict['st_customer'] = st_cus.tolist()
        
        ## Fuzzy Demand of Customers
        params_dict['fuzzy_demand'] = customer[col_list[-5: -2]].values
        
        ## Time Window of Customers
        params_dict['time_window'] = customer[col_list[3:7]].values
       
        ## Define the max capacity of Satellites as 1e+8
        params_dict['max_cap_satellite'] = [10**8] * coor_sat.shape[0]
        
        return params_dict
    
    # Define the function to upload the parameters from .dat file
    def upload_params(self):
        '''
        The style of .dat file should be follwing:
        !------------------------------------------------------------------------
        !Trucks: total #, capacity, cost per distance, fixcost 
        3,15000,1,0
        !------------------------------------------------------------------------
        !CityFreighters: max cf/sat, total#, cap, cost/dist, fixcost 
        4,4,6000,1,0
        !------------------------------------------------------------------------
        !Stores: first: depot x, y; then: satellites x,y,handlingCost,maxCap,fixCost 
        145,215  142,239,0.00,22500,0  146,208,0.00,22500,0  
        !------------------------------------------------------------------------
        !Customers: x,y,demand 
        151,264,1100  159,261,700  130,254,800  128,252,1400  163,247,2100  146,246,400  161,242,800  142,239,100  163,236,500  148,232,600  128,231,1200  156,217,1300  129,214,1300  146,208,300  164,208,900  141,206,2100  147,193,1000  164,193,900  129,189,2500  155,185,1800  139,182,700  
        '''
        params_dict = {
            'truck_depot': [],
            'Q1': [],
            'cost_ds_FE': [],
            'fix_cost_FE': [],
            'truck_satellite': [],
            'max_pass_satellite': [],
            'Q2': [],
            'cost_ds_SE': [],
            'fix_cost_SE': [],
            'coor_depot': [],
            'coor_cap_cost_satellite': [],
            'coor_demand_custo': []
            
        }
        with open(self.path, 'r') as file:
            num_line = 1
            for line in file:
                if num_line == 3:
                    result = line.split(",")
                    params_dict['truck_depot'].append(int(result[0]))
                    params_dict['Q1'].append(float(result[1]))
                    params_dict['cost_ds_FE'].append(float(result[2]))
                    params_dict['fix_cost_FE'].append(float(result[3]))
                #    num_line += 1
                if num_line == 6:
                    result = line.split(",")
                    params_dict['truck_satellite'].append(int(result[0]))
                    params_dict['max_pass_satellite'].append(int(result[1]))
                    params_dict['Q2'].append(float(result[2]))
                    params_dict['cost_ds_SE'].append(float(result[3]))
                    params_dict['fix_cost_SE'].append(float(result[4]))
                #    num_line += 1
                if num_line == 9:
                    result = line.split("  ")
                    if '\n' in result:
                        result.remove('\n')
                #    result = line.split("  ")    
                    params_dict['coor_depot'].append([float(num) for num in result[0].split(",")])
                    for i in range(1, len(result)):
                        params_dict['coor_cap_cost_satellite'].append([float(num) for num in result[i].split(",")])
                #    num_line += 1
                if num_line == 12:
                    result = line.split("  ")
                    if '\n' in result:
                        result.remove('\n')
                #    result = line.split("  ")    
                    for cus in result:
                        params_dict['coor_demand_custo'].append([float(num) for num in cus.split(",")])
                #    num_line += 1    
                
                num_line += 1
        return params_dict
                        
    ##--- Define the pre-optimized cost function        
    def calculate_cost(self, solution):
        # Calculate the total cost of the solution
        cost = 0
        route1 = solution['depot_to_sat']
        route2 = solution['sat_to_cus']
        # Iterate over each depot
        for i in range(self.n_satellite):
            path = route1[i]
            current_time = 0
            print(path)
            if len(path) > 0:
                cost += len(path) * (2 * self.depot_to_sat_distances[i] * self.unit_cost_1 + self.fixed_cost_1)
                current_time += self.depot_to_sat_distances[i] + self.st_satellite[i]
            # Cost for satellites to customers
            if len(route2[i]) > 0:
                for sub_path in route2[i]:
            #cost += self.optimize_sub_sat_solution(i, route2[i])[1]
                    if len(sub_path) > 0:
                        for j in range(len(sub_path)):
                            if j == 0:
                                current_time += self.sat_to_cus_distances[i, sub_path[j]]
                                cost += self.sat_to_cus_distances[i, sub_path[j]] * self.unit_cost_2 - time_penalty(current_time, self.time_window[sub_path[j]])
                            else:
                                current_time += self.cus_to_cus_distances[sub_path[j-1], sub_path[j]]
                                cost += self.cus_to_cus_distances[sub_path[j-1], sub_path[j]] * self.unit_cost_2 - time_penalty(current_time, self.time_window[sub_path[j]])
                        cost += self.sat_to_cus_distances[i, sub_path[-1]] * self.unit_cost_2 + self.fixed_cost_2
                   
         
        return cost
    def tabu(self, max_iterations, neighborhood_size):
        # Initialize the best solution and its cost
        best_solution = copy.deepcopy(self.solution)
        best_cost = self.cost
        
        # Set the initial neighborhood structure and its size
        neighborhood_structure = 1
        neighbourhood_size = neighborhood_size
        
        # Define the maximum number of iterations for each neighborhood structure
        max_iterations_per_structure = [2 * self.n_customers, self.n_customers]
        
        # Iterate over the specified number of iterations
        for i in range(max_iterations):
            # Initialize the current solution and its cost
            current_solution = copy.deepcopy(best_solution)
            current_cost = best_cost
            
            # Perturb the current solution using the current neighborhood structure
            if neighborhood_structure == 1:
                # Swap two randomly selected customers between two randomly selected depots
                sat1, sat2 = random.sample(range(self.n_satellite), 2)
                indices1 = []
                for a in current_solution['sat_to_cus'][sat1]:
                    indices1 += a
                indices2 = []
                for b in current_solution['sat_to_cus'][sat2]:
                    indices2 += b
                if len(indices1) > 0 and len(indices2) > 0:
                    customer1 = random.choice(indices1)
                    customer2 = random.choice(indices2)
                    indices1.remove(customer1)
                    indices1.append(customer2)
                    indices2.remove(customer2)
                    indices2.append(customer1)
                    current_solution['sat_to_cus'][sat1] = self.optimize_sub_sat_solution(sat1, [[a] for a in indices1])[0]
                    current_solution['sat_to_cus'][sat2] = self.optimize_sub_sat_solution(sat2, [[b] for b in indices2])[0]
                    current_solution = self.generate_depot_solutions(current_solution['sat_to_cus'])
            else:
                # Assign a randomly selected customer to a randomly selected depot
                sat1, sat2 = random.sample(range(self.n_satellite), 2)
                indices1 = []
                for a in current_solution['sat_to_cus'][sat1]:
                    indices1 += a
                indices2 = []
                for b in current_solution['sat_to_cus'][sat2]:
                    indices2 += b
                if len(indices1) > 0:
                    customer1 = random.choice(indices1)
                    indices1.remove(customer1)
                    indices2.append(customer1)
                    current_solution['sat_to_cus'][sat1] = self.optimize_sub_sat_solution(sat1, [[a] for a in indices1])[0]
                    current_solution['sat_to_cus'][sat2] = self.optimize_sub_sat_solution(sat2, [[b] for b in indices2])[0]
                    current_solution = self.generate_depot_solutions(current_solution['sat_to_cus'])
                if len(indices2) > 0:
                    customer2 = random.choice(indices2)
                    indices2.remove(customer2)
                    indices1.append(customer2)
                    current_solution['sat_to_cus'][sat1] = self.optimize_sub_sat_solution(sat1, [[a] for a in indices1])[0]
                    current_solution['sat_to_cus'][sat2] = self.optimize_sub_sat_solution(sat2, [[b] for b in indices2])[0]
                    current_solution = self.generate_depot_solutions(current_solution['sat_to_cus'])
            # Perform local search within the current neighborhood
            neighborhood_iterations = 0
            while neighborhood_iterations < max_iterations_per_structure[neighborhood_structure-1]:
                # Swap two randomly selected customers between two randomly selected depots
                sat1, sat2 = random.sample(range(self.n_satellite), 2)
                indices1 = []
                for a in current_solution['sat_to_cus'][sat1]:
                    indices1 += a
                indices2 = []
                for b in current_solution['sat_to_cus'][sat2]:
                    indices2 += b
                if len(indices1) > 0 and len(indices2) > 0:
                    customer1 = random.choice(indices1)
                    customer2 = random.choice(indices2)
                    indices1.remove(customer1)
                    indices1.append(customer2)
                    indices2.remove(customer2)
                    indices2.append(customer1)
                    new_solution = copy.deepcopy(current_solution['sat_to_cus'])
                    new_solution[sat1] = self.optimize_sub_sat_solution(sat1, [[a] for a in indices1])[0]
                    new_solution[sat2] = self.optimize_sub_sat_solution(sat2, [[a] for a in indices2])[0]
                    new_solution = self.generate_depot_solutions(new_solution)
                    # Calculate the cost of the new solution
                    new_cost = self.calculate_cost(new_solution)
                    
                    # If the new solution is better than the current one, update the current solution
                    if new_cost < current_cost:
                        current_solution = copy.deepcopy(new_solution)
                        current_cost = new_cost
                        
                        # If the new solution is better than the best one, update the best solution
                        if new_cost < best_cost:
                            best_solution = copy.deepcopy(new_solution)
                            best_cost = new_cost
                            
                            # Reset the neighborhood structure and size
                            neighborhood_structure = 1
                            neighbourhood_size = neighborhood_size
                
                neighborhood_iterations += 1
            
            # Update the neighborhood structure and size
            neighborhood_structure = (neighborhood_structure % 2) + 1
            neighbourhood_size += 1
        
        return best_solution, best_cost
    
    ##---------Generate Initial Solution --------------##
    def generate_init_sat_solutions(self):
    #    solutions = {'depot_to_sat': [],
    #                 'sat_to_cus': []}
        solutions = []
        ### Get the solution of satellites to customers first
        sat_cap1 = np.zeros(self.n_satellite)
        for j in range(self.n_satellite):
            solutions.append([])
        for i in range(self.n_customers): 
            depot_to_cus = self.sat_to_cus_distances[:, i] + self.depot_to_sat_distances
            t = np.array([time_penalty(depot_to_cus[i], self.time_window[i]) for i in range(self.n_satellite)])
            idx = np.argmax(t)
            solutions[idx].append([i])
            sat_cap1[idx] += self.expected_demand[i]
        return solutions, sat_cap1    
        ### Get the solution of depot to satellites
    
    def optimize_double_sub_sat_solution(self, sat_idx, sub_sub_solution):
        if len(sub_sub_solution) == 0:
            return sub_sub_solution, 0
        
        ## Define the cost function of sub_sub_solution
        def sub_cost(current_time, sub_path):
            cost = 0
            current_time1 = current_time
            if len(sub_path) == 0:
                return 0
            for j in range(len(sub_path)):
                if j == 0:
                    current_time1 += self.sat_to_cus_distances[sat_idx, sub_path[j]]
                    cost += self.sat_to_cus_distances[sat_idx, sub_path[j]] * self.unit_cost_2 - time_penalty(current_time1, self.time_window[sub_path[j]])
                else:
                    current_time1 += self.cus_to_cus_distances[sub_path[j-1], sub_path[j]]
                    cost += self.cus_to_cus_distances[sub_path[j-1], sub_path[j]] * self.unit_cost_2 - time_penalty(current_time1, self.time_window[sub_path[j]])
            cost += self.sat_to_cus_distances[sat_idx, sub_path[-1]] * self.unit_cost_2 + self.fixed_cost_2
            return cost
        
        max_iteration = len(sub_sub_solution) ** 2
        best_solution = sub_sub_solution
        timestart = self.depot_to_sat_distances[sat_idx]
        best_cost = sub_cost(timestart, sub_sub_solution)
        ## Implement Tabu Search
        if len(sub_sub_solution) > 1:
            for iter in range(max_iteration):
                idx1, idx2 = np.random.randint(0, len(sub_sub_solution), 2)
                neighbor_solution = best_solution.copy()
                #Swap
                neighbor_solution[idx1], neighbor_solution[idx2] = neighbor_solution[idx2], neighbor_solution[idx1]
                if sub_cost(timestart, neighbor_solution) < best_cost:
                    best_solution = neighbor_solution
                    best_cost = sub_cost(timestart, neighbor_solution)
            
        return [best_solution, best_cost]
        
    def optimize_sub_sat_solution(self, sat_idx, sub_solution):
        if len(sub_solution) == 0:
            return sub_solution, 0
        best_solution_cost = [self.optimize_double_sub_sat_solution(sat_idx, sub_sub_solution) for sub_sub_solution in sub_solution]
        best_solution = [sol[0] for sol in best_solution_cost]
        
        best_cost = sum([sol[1] for sol in best_solution_cost])
        max_iteration = sum([len(sub_sub_solution) for sub_sub_solution in sub_solution]) ** 2
        ## Implement Tabu Search
        step = 0
        while len(best_solution) > 1 and step < max_iteration:
            idx1, idx2 = np.random.randint(0, len(best_solution), 2)
            permutation_prob = random.random()
            current_solution = best_solution.copy()
            if permutation_prob >= 0.4 and len(current_solution[idx2]) > 0:
                transfer_idx = np.random.randint(0, len(current_solution[idx2]))
                if sum([self.expected_demand[i] for i in current_solution[idx1]]) + self.expected_demand[current_solution[idx2][transfer_idx]] <= self.vehicle2_cap:
                    current_solution[idx1].append(current_solution[idx2][transfer_idx])
                    current_solution[idx2].remove(current_solution[idx2][transfer_idx])
                if len(current_solution[idx2]) == 0:
                    current_solution.remove(current_solution[idx2])
            if permutation_prob <= 0.4 and len(current_solution[idx2]) > 0 and len(current_solution[idx1]) > 0:
                transfer_idx1, transfer_idx2 = np.random.randint(0, len(current_solution[idx1])), np.random.randint(0, len(current_solution[idx2]))
                a, b = current_solution[idx1][transfer_idx1], current_solution[idx2][transfer_idx2]
                if sum([self.expected_demand[i] for i in current_solution[idx1]]) - self.expected_demand[a] + self.expected_demand[b] <= self.vehicle2_cap and sum([self.expected_demand[i] for i in current_solution[idx2]]) + self.expected_demand[a] - self.expected_demand[b] <= self.vehicle2_cap:
                    ##SWAP
                    current_solution[idx1].remove(a)
                    current_solution[idx1].append(b) 
                    #-----
                    current_solution[idx2].remove(b) 
                    current_solution[idx2].append(a) 
            current_solution_cost = [self.optimize_double_sub_sat_solution(sat_idx, sub_sub_solution) for sub_sub_solution in current_solution]
            if sum([sol[1] for sol in current_solution_cost]) < best_cost:
                best_solution = [sol[0] for sol in current_solution_cost]
                best_cost = sum([sol[1] for sol in current_solution_cost])
            step += 1
        return best_solution, best_cost
            
    ## Define the depot to satellite generation function with optimized route2
    def generate_depot_solutions(self, solutions):
        solution = {'depot_to_sat': [],
                     'sat_to_cus': []}    
        alter_solution = solutions
        solution['sat_to_cus'] = [self.optimize_sub_sat_solution(i, alter_solution[i])[0] for i in range(self.n_satellite)]
        ## get the required amount of goods in each satellite
        required_amount = np.zeros(self.n_satellite)
        for i in range(self.n_satellite):
            if len(solution['sat_to_cus'][i]) > 0:
                for sub in solution['sat_to_cus'][i]:
                    for sub1 in sub:
                        required_amount[i] += self.expected_demand[sub1]
                        
        ## Fill the depot to satellite solutions
        carry_mount = self.vehicle1_cap
        for i in range(self.n_satellite):
            if required_amount[i] > 0:
                n_move = int((required_amount[i]-1) / carry_mount) + 1
                i_path = [-1] * n_move
                solution['depot_to_sat'].append(i_path)
            else: 
                solution['depot_to_sat'].append([])
                     
        return solution
    
    
def scatter_search_vns(twoevrp, max_iterations, num_solutions, neighborhood_size=2):
    # Generate a diverse set of initial solutions using scatter search
    initial_s = twoevrp.generate_init_sat_solutions()[0]
    initial_solutions = twoevrp.generate_depot_solutions(initial_s)

    # Initialize the best solution and its cost
    best_solution = initial_solutions
    best_cost = twoevrp.calculate_cost(best_solution)

    for i in range(num_solutions):
        improved_solution, improved_cost = twoevrp.tabu(max_iterations, neighborhood_size)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_solution = improved_solution
            
    return best_solution        

##Import Library
import os

def output_csv_plot(dat_file,max_iterations, neighborhood_size):
    #Delete All old file from CSV Folder
    for file in os.listdir('CSV'):
        os.remove(os.path.join('CSV', file))
    path = os.path.join('excel_params', dat_file)
    twoecvrp = TwoECVrp(path)
    num_solutions = twoecvrp.n_satellite + 1
    best_sol = scatter_search_vns(twoecvrp, max_iterations, num_solutions, neighborhood_size)
    best_cost = twoecvrp.calculate_cost(best_sol)
    
    data = {'Total_Cost': [best_cost], 
            'First_Route': [best_sol['depot_to_sat']], 
            'Second_Route':[best_sol['sat_to_cus']],
            'Expected_demand': [twoecvrp.expected_demand],
            'Real_demand': [twoecvrp.real_demand]}
                
    df = pd.DataFrame(data)
    print('-1 is the depot, other indice are satellite')
    title_ = str(dat_file) + str('_') + str(max_iterations) + str('_') + str(num_solutions) + str('_') + 'tabu' + '.csv'
    final_path = os.path.join('CSV', title_)
    df.to_csv(final_path, index = False)       
     
e = TwoECVrp(path)                

