# Import necessary libraries
import numpy as np
import random
import copy
import scipy
import pandas as pd
import sklearn
## Define Path for Code Testing
#path = "E:/Project/Upwork/Satellite_Algorithm/data/Set2a_E-n22-k4-s8-14.dat"   
# Define the 2E-VRP class
class TwoECVrp:
    
    def __init__(self, path):
        # Set the number of customers and depots
        self.path = path
        params_dict = self.upload_params()
        self.n_customers = len(params_dict['coor_demand_custo'])
        self.n_satellite = len(params_dict['coor_cap_cost_satellite'])
        
        # Generate random demand for each customer
        self.demand = []
        self.cus_sumary = params_dict['coor_demand_custo']
        for cus in self.cus_sumary:
            self.demand.append(cus[2:])
        self.demand = np.array(self.demand)
          
        
        # Generate random distances between all nodes
        self.depot_to_sat_distances = np.zeros(self.n_satellite) # Distance between depot and satellite
        self.sat_to_cus_distances = np.zeros((self.n_satellite, self.n_customers)) # Distance between satellite and customers
        self.sat_to_sat_distances = np.zeros((self.n_satellite, self.n_satellite))
        # Set the capacity of satellite (But Optional)
        self.sat_cap = np.zeros(self.n_satellite)
        
        # Set the commissions of each satellite
        self.hs = np.zeros(self.n_satellite)
        for i in range(self.n_satellite):
            self.sat_cap[i] = params_dict['coor_cap_cost_satellite'][i][3]
            self.hs[i] = params_dict['coor_cap_cost_satellite'][i][4]
            coor_sati = np.array(params_dict['coor_cap_cost_satellite'][i][0:2])
            coor_depot = np.array(params_dict['coor_depot'][0])
            self.depot_to_sat_distances[i] = sum((coor_sati - coor_depot) ** 2)
            
            for j in range(self.n_customers):
                coor_cus = np.array(params_dict['coor_demand_custo'][j][0:2])
                self.sat_to_cus_distances[i, j] = sum((coor_sati - coor_cus) ** 2)
            
            for s in range(self.n_satellite):
                coor_sats = np.array(params_dict['coor_cap_cost_satellite'][s][0:2])
                self.sat_to_sat_distances[i, s] = sum((coor_sats - coor_sati) ** 2)    
                
        ## Define the distance between cus to cus
        self.cus_to_cus_distances = np.zeros((self.n_customers, self.n_customers))
        for i in range(self.n_customers):
            for j in range(self.n_customers):
                coori = np.array(params_dict['coor_demand_custo'][i][0:2])
                coorj = np.array(params_dict['coor_demand_custo'][j][0:2])
                self.cus_to_cus_distances[i, j] = sum((coori - coorj)**2)

        
        # Set the capacity of the vehicles
        self.vehicle1_cap = params_dict['Q1'][0]
        self.vehicle2_cap = params_dict['Q2'][0] 
        
        # Set the number of vehicles available at each depot
        self.vehicle1_num = params_dict['truck_depot'][0]
        self.vehicle2_num = params_dict['truck_satellite'][0]
        
        # Set the limit pass of each satellite
        self.sat_pass = params_dict['max_pass_satellite'][0]
        
        # Set the cost the unit
        self.unit_cost_1 = params_dict['cost_ds_FE'][0]
        self.unit_cost_2 = params_dict['cost_ds_SE'][0]
        
        # Initialize the solution with a random assignment of customers to depots
        self.sat_solution = self.generate_init_sat_solutions()
        self.solution = self.generate_depot_solutions(self.sat_solution)
        
        # Calculate the cost of the initial solution
        self.cost = self.calculate_cost(self.solution)
    
        
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
                        
        
    def calculate_cost(self, solution):
        # Calculate the total cost of the solution
        cost = 0
        route1 = solution['depot_to_sat']
        route2 = np.array(solution['sat_to_cus'])
        # Iterate over each depot
        for route in route1:
            l = len(route)
            cost += self.depot_to_sat_distances[route[1]] + self.hs[route[1]]
            cost += self.depot_to_sat_distances[route[l-2]] + self.hs[route[l-2]]
            if l > 3:
                for i in range(1, l-2):
                    cost += self.sat_to_sat_distances[route[i], route[i+1]] + self.hs[route[i]]
        cost *= self.unit_cost_1
            # Get the indices of the customers assigned to this depot
        for s in range(self.n_satellite):
            cust = list(np.where(route2 == s))[0]
            l = len(cust)
            if l == 0:
                continue
            else:
                cost += self.sat_to_cus_distances[s, cust[0]] * self.unit_cost_2+ self.hs[s]
                cost += self.sat_to_cus_distances[s, cust[l-1]] * self.unit_cost_2+ self.hs[s]
                if l > 1:
                    for c in range(l-1):
                        cost += self.cus_to_cus_distances[cust[c], cust[c+1]] * self.unit_cost_2
        return cost
               
    def vns_2(self, max_iterations, neighborhood_size):
        # Initialize the best solution and its cost
        best_solution = copy.deepcopy(self.solution)
        best_cost = self.cost
        
        # Set the initial neighborhood structure and its size
        neighborhood_structure = 1
        neighborhood_size = neighborhood_size
        
        # Define the maximum number of iterations for each neighborhood structure
        max_iterations_per_structure = [20, 10]
        
        # Iterate over the specified number of iterations
        for i in range(max_iterations):
            # Initialize the current solution and its cost
            current_solution = copy.deepcopy(best_solution)
            current_cost = best_cost
            
            # Perturb the current solution using the current neighborhood structure
            if neighborhood_structure == 1:
                # Swap two randomly selected customers between two randomly selected depots
                sat1, sat2 = random.sample(range(self.n_satellite), 2)
                indices1 = list(np.where(np.array(current_solution['sat_to_cus']) == sat1))[0]
                indices2 = list(np.where(np.array(current_solution['sat_to_cus']) == sat2))[0]
                if len(indices1) > 0 and len(indices2) > 0:
                    customer1 = random.choice(indices1)
                    customer2 = random.choice(indices2)
                    current_solution['sat_to_cus'][customer1] = sat2
                    current_solution['sat_to_cus'][customer2] = sat1
                    current_solution = self.generate_depot_solutions(current_solution['sat_to_cus'])
            else:
                # Assign a randomly selected customer to a randomly selected depot
                customer = random.choice(range(self.n_customers))
                sat = random.choice(range(self.n_satellite))
                current_solution['sat_to_cus'][customer] = sat
                current_solution = self.generate_depot_solutions(current_solution['sat_to_cus'])
            # Perform local search within the current neighborhood
            neighborhood_iterations = 0
            while neighborhood_iterations < max_iterations_per_structure[neighborhood_structure-1]:
                # Swap two randomly selected customers between two randomly selected depots
                sat1, sat2 = random.sample(range(self.n_satellite), 2)
                indices1 = list(np.where(np.array(current_solution['sat_to_cus']) == sat1))[0]
                indices2 = list(np.where(np.array(current_solution['sat_to_cus']) == sat2))[0]
                if len(indices1) > 0 and len(indices2) > 0:
                    customer1 = random.choice(indices1)
                    customer2 = random.choice(indices2)
                    new_solution = copy.deepcopy(current_solution['sat_to_cus'])
                    new_solution[customer1] = sat2
                    new_solution[customer2] = sat1
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
                            neighborhood_size = neighborhood_size
                
                neighborhood_iterations += 1
            
            # Update the neighborhood structure and size
            neighborhood_structure = (neighborhood_structure % 2) + 1
            neighborhood_size += 1
        
        return best_solution, best_cost
    
    def generate_init_sat_solutions(self):
    #    solutions = {'depot_to_sat': [],
    #                 'sat_to_cus': []}
        solutions = []
        total_demands = sum(self.demand)
        ### Get the solution of satellites to customers first
        sat_cap1 = np.zeros(self.n_satellite)
        for i in range(self.n_customers):
            sat_ind = 0
            #sat_cap_filt = sat_cap1[np.where(sat_cap1 >= self.demand[i])]
            sat_cap_filt_ind = list(np.where(sat_cap1 <= self.sat_cap - self.demand[i]))[0]
            rand_ind = random.randint(0, len(sat_cap_filt_ind)-1)
         #   min_ind = np.argmin(self.sat_to_cus_distances[sat_cap_filt_ind.tolist(), i])
            solutions.append(sat_cap_filt_ind[rand_ind])
            sat_cap1[sat_cap_filt_ind[rand_ind]] += self.demand[i]
        return solutions    
        ### Get the solution of depot to satellites
    def generate_depot_solutions(self, solutions):
        solution = {'depot_to_sat': [],
                     'sat_to_cus': []}    
        solution['sat_to_cus'] = solutions
        carry_mount = self.vehicle1_cap
        sat_cap = np.zeros(self.n_satellite)
        for i in range(self.n_satellite):
            ind = list(np.where(np.array(solutions) == i))[0]
            if len(ind) > 0:
                sat_cap[i] = sum(self.demand[ind])
        sat_index = 0
        while sat_index < self.n_satellite and sat_cap[self.n_satellite-1] > 0:
            sat_list = [-1]
            while carry_mount > 0 and sat_index < self.n_satellite:
                sat_list.append(sat_index)
                if sat_cap[sat_index] > carry_mount:
                    sat_cap[sat_index] -= carry_mount 
                    carry_mount = 0
                    sat_list.append(-1) 
                if sat_cap[sat_index] == carry_mount:
                    sat_index += 1
                    carry_mount = 0
                    sat_list.append(-1)
                if sat_cap[sat_index] < carry_mount:
                    if sat_index < self.n_satellite - 1:
                        carry_mount -= sat_cap[sat_index]
                    #    sat_list.append(sat_index)
                        sat_index += 1
                    else: 
                        sat_list.append(-1)
                        sat_index = self.n_satellite
                        carry_mount = 0    
            solution['depot_to_sat'].append(sat_list)
            sat_list = [-1]
            carry_mount = self.vehicle1_cap                     
                 
        return solution
    
    
def scatter_search_vns(twoevrp, max_iterations, num_solutions, neighborhood_size=2):
    # Generate a diverse set of initial solutions using scatter search
    initial_s = twoevrp.generate_init_sat_solutions()
    initial_solutions = twoevrp.generate_depot_solutions(initial_s)

    # Initialize the best solution and its cost
    best_solution = initial_solutions
    best_cost = twoevrp.calculate_cost(best_solution)

    for i in range(num_solutions):
        improved_solution, improved_cost = twoevrp.vns_2(max_iterations, neighborhood_size)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_solution = improved_solution
            
    return best_solution        

##Import Library
import os
def output_csv(dat_file,max_iterations, num_solutions, neighborhood_size):
    #Delete All old file from CSV Folder
    for file in os.listdir('CSV'):
        os.remove(os.path.join('CSV', file))
    path = os.path.join('data', dat_file)
    twoecvrp = TwoECVrp(path)
    best_sol = scatter_search_vns(twoecvrp,max_iterations, num_solutions, neighborhood_size)
    best_cost = twoecvrp.calculate_cost(best_sol)
    
    data = {'Total_Cost': [best_cost], 
            'First_Route': [best_sol['depot_to_sat']], 
            'Second_Route':[best_sol['sat_to_cus']]}
                
    df = pd.DataFrame(data)
    print('-1 is the depot, other indice are satellite')
    title_ = str(dat_file) + str('_') + str(max_iterations) + str('_') + str(num_solutions) + str('_') + str(neighborhood_size) + '.csv'
    final_path = os.path.join('CSV', title_)
    df.to_csv(final_path, index = False)        
#e = TwoECVrp(path)                