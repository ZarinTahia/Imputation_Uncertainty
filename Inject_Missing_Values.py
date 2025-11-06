
import numpy as np
import pandas as pd
import random
from collections import defaultdict

class Inject_Missing_Values:
    def __init__(self):
        self.missing_log = defaultdict(set)
        self.all_dependent_cells_mar = set()
        self.all_dependent_cells_mnar = set()
        self.missing_indices_mar = set()
        self.influencing_indices_mar = set()
        self.influence_map_mar = {}
        

    def MCAR(self, data: pd.DataFrame, selected_columns: list = None, missing_rate: int = 10, seed: int = 42):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas DataFrame')
        if not (0 <= missing_rate < 100):
            raise ValueError('Missing rate must be between 0 and 100')
        
        np.random.seed(seed)
        X = data.copy()
        total_rows, total_cols = X.shape
        num_missing = round(total_rows * total_cols * (missing_rate / 100))
        
        all_indices = (
            [(row, X.columns.get_loc(col)) for col in selected_columns for row in range(total_rows)]
            if selected_columns else [(row, col) for row in range(total_rows) for col in range(total_cols)]
        )
        
        pos_miss = random.sample(all_indices, min(num_missing, len(all_indices)))
        for row, col in pos_miss:
            X.iat[row, col] = np.nan
            self.missing_log[(row, col)].add("MCAR")
        
        missing_log_formatted = {f"{row},{col}" : ",".join(types) for (row, col), types in self.missing_log.items()}
        return X, missing_log_formatted

    def MAR(self, X, dependencies=None, missing_rate=15, random_seed=42):

      

        np.random.seed(random_seed)
        data = X.copy()
        total_cells = data.size
       
        n_missing = int(total_cells * (missing_rate / 100))
      

        if dependencies is None:
            dependencies = {col: {
                "influencers": [random.choice([c for c in data.columns if c != col])],
                "condition": lambda row, chosen_col=random.choice(data.columns): row[chosen_col] > data[chosen_col].mean()
            } for col in data.columns}

        self.influence_map_mar = {col: [] for col in data.columns}
        eligible_indices_mar = []
        
        for target, dependency in dependencies.items():
        
            influencers = dependency.get("influencers", [])
            condition = dependency.get("condition", lambda row: False)
            probability_function = dependency.get("probability", lambda row: 1.0)

            eligible_rows = data.index[data.apply(condition, axis=1)]
            for row_index in eligible_rows:
                prob = probability_function(data.loc[row_index])
                eligible_indices_mar.append((row_index, data.columns.get_loc(target), prob))
               
            
            for i in influencers:
                self.influence_map_mar[i].append(target)
      
        #eligible_indices_mar.sort(key=lambda x: x[2], reverse=True)
        #mar_missing_indices = eligible_indices_mar[:n_missing]

        # Extract only the row and column indices, ignoring the probability field
       

# Extract row, column indices separately
        indices = [(x[0], x[1]) for x in eligible_indices_mar]  # Extract only (row, col)
        
        
        probabilities = [x[2] for x in eligible_indices_mar]  # Extract probability weights

        # Normalize probabilities
        total_prob = sum(probabilities)
        normalized_probabilities = [p / total_prob for p in probabilities]

        # Create index positions (0, 1, 2, ..., len(indices)-1)
        index_positions = np.arange(len(indices))
       

        # Select unique positions based on probability
        #print(n_missing)
        # Ensure n_missing does not exceed the total available indices
        n_missing = min(n_missing, len(indices))  # Prevents oversampling
       

        # Select unique positions based on probability
        selected_positions = np.random.choice(index_positions, size=n_missing, replace=False, p=normalized_probabilities)
       
        # Map back to (row, col) tuples
        mar_missing_indices = [indices[i] for i in selected_positions]


        #print(self.influence_map_mar)

        for row, col in mar_missing_indices:
            data.iat[row, col] = np.nan
            self.missing_log[(row, col)].add("MAR")
            self.missing_indices_mar.add((row, col))
        
        for row, col in self.missing_indices_mar:

            influencing_col = data.columns[col]
            #print("influencing_col",influencing_col)
            influenced_cols = self.influence_map_mar.get(influencing_col, [])
            
            influenced_col_indices = [data.columns.get_loc(col_name) for col_name in influenced_cols] #only the missing influencers
        
            self.all_dependent_cells_mar.update((row, col_idx) for col_idx in influenced_col_indices)
            #print(self.all_dependent_cells_mar)
        
        for row, col in self.all_dependent_cells_mar:
            if pd.isna(data.iat[row, col]):
                self.missing_log[(row, col)].add("MNAR")
        
        missing_log_formatted = {f"{row},{col}" : ",".join(types) for (row, col), types in self.missing_log.items()}
        return data, missing_log_formatted

    def MNAR(self, X, dependencies, missing_rate, random_seed=42):

        np.random.seed(random_seed)
        data = X.copy()
        total_cells = data.size
        n_missing = int(total_cells * (missing_rate / 100))
        #print(n_missing)

        if dependencies is None:
            return data

        eligible_indices_mnar = []
        
        for target, dependency in dependencies.items():
            condition = dependency.get("condition", lambda row: False)
            probability_function = dependency.get("probability", lambda row: 1.0)

            eligible_rows = data.index[data.apply(condition, axis=1)]
            for row_index in eligible_rows:
                prob = probability_function(data.loc[row_index])
                eligible_indices_mnar.append((row_index, data.columns.get_loc(target), prob))
        
        #eligible_indices_mnar.sort(key=lambda x: x[2], reverse=True)
        #mnar_missing_indices = eligible_indices_mnar[:n_missing]

       # Extract only the row and column indices, ignoring the probability field
        

# Extract row, column indices separately
        indices = [(x[0], x[1]) for x in eligible_indices_mnar]  # Extract only (row, col)
        probabilities = [x[2] for x in eligible_indices_mnar]  # Extract probability weights

        # Normalize probabilities
        total_prob = sum(probabilities)
        normalized_probabilities = [p / total_prob for p in probabilities]

        # Create index positions (0, 1, 2, ..., len(indices)-1)
        index_positions = np.arange(len(indices))
        #print(n_missing)
        # Ensure n_missing does not exceed the total available indices
        n_missing = min(n_missing, len(indices))  # Prevents oversampling
        #print(n_missing)

        # Select unique positions based on probability
        selected_positions = np.random.choice(index_positions, size=n_missing, replace=False, p=normalized_probabilities)

        # Map back to (row, col) tuples
        mnar_missing_indices = [indices[i] for i in selected_positions]



        for row, col in mnar_missing_indices:
                data.iat[row, col] = np.nan
                self.missing_log[(row, col)].add("MNAR")
                
            
            # Check if any influencing MAR indices became missing, then mark its dependent cells as MNAR
        for row, col in mnar_missing_indices:

            influencing_col = data.columns[col]
            #print("influencing_col",influencing_col)
            influenced_cols = self.influence_map_mar.get(influencing_col, [])
            
            influenced_col_indices = [data.columns.get_loc(col_name) for col_name in influenced_cols] #only the missing influencers
        
            self.all_dependent_cells_mnar.update((row, col_idx) for col_idx in influenced_col_indices)
            #print(self.all_dependent_cells_mar)
        
        for row, col in self.all_dependent_cells_mnar:
            if pd.isna(data.iat[row, col]):
                self.missing_log[(row, col)].add("MNAR")
        missing_log_formatted = {f"{row},{col}" : ",".join(types) for (row, col), types in self.missing_log.items()}
        return data, missing_log_formatted






'''''
import numpy as np
import pandas as pd
import warnings

import numpy as np
import pandas as pd
import random
from collections import defaultdict

class Inject_Missing_Values:

    missing_log = defaultdict(set)
    all_dependent_cells_mar =[]   
    influencing_indices_mar=[]        # Iterate over dependencies to introduce missingness
    influence_map_mar= {}
    


    def MCAR(self, X: pd.DataFrame, selected_columns: list = None,missing_rate: int = 10, seed: int = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas DataFrame')

        if missing_rate >= 100:
            raise ValueError('Missing Rate is too high, the whole dataset will be deleted!')
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        self.X = X
        self.dataset = self.X.copy()
        self.missing_rate = missing_rate
        self.mcar_missing_indices = []

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        """
        Function to randomly generate missing data in the entire dataset.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MCAR mechanism.
        """
       
        mr = self.missing_rate / 100
       

        # Convert the dataset values to float to allow NaN
        
        total_rows, total_cols = self.X.shape
        num_missing = round(total_rows * total_cols * mr)

        # Select random positions for missingness
        if selected_columns is None:
            # Randomly select positions across the whole dataset
            #num_missing = round(total_rows * total_cols * mr)
            all_indices = [(row, col) for row in range(total_rows) for col in range(total_cols)]
            pos_miss = random.sample(all_indices, num_missing)
        else:
            # Apply missingness only to selected columns
            all_indices = [(row, self.X.columns.get_loc(col)) for col in selected_columns for row in range(total_rows)]
            #num_missing = round(len(all_indices) * mr)
            pos_miss = random.sample(all_indices, num_missing)

        # Track missing indices
        for row, col in pos_miss:
            self.dataset.iloc[row, col] = np.nan
            self.mcar_missing_indices.append((row, col))
            self.missing_log[(row, col)].add("MCAR")
        
        missing_log_formatted = {f"{row},{col}" : ",".join(types) for (row, col), types in self.missing_log.items()}
        # Flatten the array, insert NaN, and reshape
        

        return self.dataset, missing_log_formatted


    def MAR(self,data, dependencies=None, missing_rate=15, random_seed=42):
        """
        Generate MAR missingness in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset, which may already have missing values.
        missing_rate : float
            Percentage of new missing values to introduce, calculated on non-missing data.
        dependencies : dict, optional
            Dependencies for MAR missingness. Keys are target columns, values are:
            {"influencers": list of influencing columns, "condition": callable condition}.
            Example:
            {
                "B": {"influencers": ["A"], "condition": lambda row: row["A"] > 80.0},
                "C": {"influencers": ["A", "B"], "condition": lambda row: row["A"] + row["B"] > 90.0}
            }
        random_seed : int, optional
            Seed for reproducibility.

        Returns:
        --------
        data_with_missing : pd.DataFrame
            Dataset with MAR missingness introduced.
        new_missing_indices : list
            List of tuples (row, column) representing new missing cells.
        """
       
        np.random.seed(random_seed)

        # Copy the data to avoid modifying the original dataset
        data_mar = data.copy()

        # Find indices of already missing values
        #existing_missing = data_mar.isna()

        # Initialize list to track new missing indices
        mar_missing_indices = []
        #influencing_cells = set()

        #calculating total number of cells
        total_cells = data_mar.size

        #total_non_missing = total_cells - existing_missing.sum().sum()
        n_missing = int(total_cells * (missing_rate / 100))
        

        # If no dependencies are provided, apply default behavior
        if dependencies is None:
            #print("no condition given")
            dependencies = {}
            for col in data_mar.columns:
                influencers = [c for c in data_mar.columns if c != col]
                chosen_influencer = random.choice(influencers)  # Randomly select an influencer
                dependencies[col] = {
                    "influencers": [chosen_influencer],
                    "condition": lambda row, chosen_col=chosen_influencer: row[chosen_col] > data_mar[chosen_col].mean()
                
                } 

               
        self.influence_map_mar= {col: [] for col in data_mar.columns}
        eligible_indices_mar = []
        probability_map = []
         
        for target, dependency in dependencies.items():
            #print("target",target)
            influencers = dependency.get("influencers", [])
            chosen_influencer = random.choice(influencers) #if influencers is None
            condition = dependency.get("condition", lambda row, chosen_col=chosen_influencer: row[chosen_col] > data_mar[chosen_col].mean())
            probability_function = dependency.get("probability", lambda row: 1.0)
                
            eligible_rows_mask = data_mar.apply(condition, axis=1).astype(bool)
            eligible_rows = data_mar.index[eligible_rows_mask]
            #print(eligible_rows)
            for row_index in eligible_rows:
                prob = probability_function(data_mar.loc[row_index])
                eligible_indices_mar.append((row_index, data_mar.columns.get_loc(target),prob))
                probability_map.append((row_index, data_mar.columns.get_loc(target), prob))
                
            for i in influencers:
                self.influence_map_mar[i].append(target)
                
            if eligible_rows.empty:
                continue
                
            self.influencing_indices_mar.extend([(row, data_mar.columns.get_loc(inf)) for row in eligible_rows for inf in influencers])
        
        probability_map.sort(key=lambda x: x[2], reverse=True)
        eligible_indices_mar.sort(key=lambda x: x[2], reverse=True)
        #print(eligible_indices_mar)
        #print(probability_map)
        #print(self.influencing_indices_mar)
        #eligible_indices =  sorted(eligible_indices, key=lambda x: (x[0], x[1]))
        #influencing_indices = sorted(influencing_indices, key=lambda x: (x[0], x[1]))           
        #print("eligible_indices",eligible_indices)
        #print("influencing_indices",influencing_indices)
        #print("influence_map",influence_map)

        while eligible_indices_mar and len(mar_missing_indices) < n_missing:
        
            #target_index = random.choice(eligible_indices_mar) #randomly selected index
            selected_idx = np.random.choice(len(eligible_indices_mar), p=[prob / sum(x[2] for x in eligible_indices_mar) for _, _, prob in eligible_indices_mar]) # Randomly pick from top 3 highest probability
            target_index = eligible_indices_mar.pop(selected_idx)
            
            #print((target_index[0],target_index[1])) 
            #print("target index",target_index)
            # Check if the selected index is an influencing cell for any other index
            if ((target_index[0],target_index[1])) in self.influencing_indices_mar:
                # Find dependent indices for this influencing cell
                #print("Influencing cell")

                row, col, prob = target_index  # Unpack the target index
                influencing_col = data_mar.columns[col]  # Find the column name of the influencing cell

                # Get the columns influenced by this column using the influence map
                influenced_columns = self.influence_map_mar.get(influencing_col, [])

                # Convert influenced column names to their corresponding column indices
                influenced_col_indices = [data_mar.columns.get_loc(col_name) for col_name in influenced_columns]

                # Combine the row with the influenced column indices to form dependent cell indices
                dependent_cells = [(row, col_idx) for col_idx in influenced_col_indices]
                self.all_dependent_cells_mar.append(dependent_cells)
                print("dep",dependent_cells)
                
                
          
            data_mar.iloc[target_index[0], target_index[1]] = None
            mar_missing_indices.append((target_index[0], target_index[1]))
            #eligible_indices_mar.remove(target_index)

        for sublist in self.all_dependent_cells_mar:  # Loop through each sublist
            for [row, col] in sublist:  # Loop through tuples inside each sublist
                value = data_mar.iloc[row, col]  # Get the value from the dataframe
                if pd.isna(value):  # Check if the value is NaN (missing)
                    print(f"Missing at ({row}, {col})")
                    self.missing_log[(row, col)].add("MNAR")  # Log as MNAR



        mar_missing_indices = sorted(mar_missing_indices, key=lambda x: (x[0], x[1]))
        
        for row, col,  in mar_missing_indices:
            self.missing_log[(row, col)].add("MAR") 

        missing_log_formatted = {f"{row},{col}" : ",".join(types) for (row, col), types in self.missing_log.items()}
               
        return data_mar, missing_log_formatted

    
    def MNAR(self,data, dependencies, missing_rate, random_seed=42):
        """
        Generate MNAR missingness in the dataset in a simplified manner.
        """
        np.random.seed(random_seed)
        data_mnar = data.copy()
        total_cells = data_mnar.size
        n_missing = int(total_cells * (missing_rate / 100))

        if dependencies is None:
            #print("No condition given. Provide a dependency structure for MNAR.")
            return data_mnar
        
        eligible_indices_mnar = []
        eligible_indices_mnar2 = []
        mnar_missing_indices = []
        probability_map = []
        for target, dependency in dependencies.items():
            condition = dependency.get("condition", lambda row: False)  # User-defined condition
            probability_function = dependency.get("probability", lambda row: 1.0)
            #eligible_rows = data_mnar.loc[data_mnar.apply(condition, axis=1)].index  # Get eligible row indices

            eligible_rows_mask = data_mnar.apply(condition, axis=1).astype(bool)
            eligible_rows = data_mnar.index[eligible_rows_mask]
                
            for row_index in eligible_rows:
                prob = probability_function(data_mnar.loc[row_index])
                eligible_indices_mnar.append((row_index, data_mnar.columns.get_loc(target),prob))
                probability_map.append((row_index, data_mnar.columns.get_loc(target), prob))

            
        probability_map.sort(key=lambda x: x[2], reverse=True)
        eligible_indices_mnar.sort(key=lambda x: x[2], reverse=True)

        #print(probability_map)
        #print(eligible_indices_mnar)

        while eligible_indices_mnar and len(mnar_missing_indices) < n_missing:''
        
            selected_idx = np.random.choice(len(eligible_indices_mnar), p=[prob / sum(x[2] for x in eligible_indices_mnar) for _, _, prob in eligible_indices_mnar]) #randomly selected index
            target_index = eligible_indices_mnar.pop(selected_idx)
            #print("target index",target_index)
            # Check if the selected index is an influencing cell for any other index
            if ((target_index[0],target_index[1])) in self.influencing_indices_mar:
                # Find dependent indices for this influencing cell
                #print("Influencing cell")

                row, col,prob = target_index  # Unpack the target index
                influencing_col = data_mnar.columns[col]  # Find the column name of the influencing cell

                # Get the columns influenced by this column using the influence map
                influenced_columns = self.influence_map_mar.get(influencing_col, [])

                # Convert influenced column names to their corresponding column indices
                influenced_col_indices = [data_mnar.columns.get_loc(col_name) for col_name in influenced_columns]

                # Combine the row with the influenced column indices to form dependent cell indices
                dependent_cells = [(row, col_idx) for col_idx in influenced_col_indices]
                self.all_dependent_cells_mar.append(dependent_cells)
                print(dependent_cells)

            data_mnar.iloc[target_index[0], target_index[1]] = None
            mnar_missing_indices.append(((target_index[0],target_index[1])))
            #eligible_indices_mnar.remove(target_index)

        for sublist in self.all_dependent_cells_mar:  # Loop through each sublist
            for [row, col] in sublist:  # Loop through tuples inside each sublist
                value = data_mnar.iloc[row, col]  # Get the value from the dataframe
                if pd.isna(value):  # Check if the value is NaN (missing)
                    #print(f"Missing at ({row}, {col})")
                    self.missing_log[(row, col)].add("MNAR")
        
        mnar_missing_indices = sorted(mnar_missing_indices, key=lambda x: (x[0], x[1]))
        
        for row, col in mnar_missing_indices:
            self.missing_log[(row, col)].add("MNAR") 

        missing_log_formatted = {f"{row},{col}" : ",".join(types) for (row, col), types in self.missing_log.items()}


        return data_mnar, missing_log_formatted 
        
'''''