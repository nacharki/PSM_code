
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from tableone import TableOne


class PropensityMatching:
    def __init__(self, Treatment_W, Covariates_X, outcome_Y, 
                 model = "LogisticRegression", method = 'min',
                 threshold =  5e-3 , max_rand = 10, nmatches = 1):
        """ 
        Basic options
            - model: the average of the outcome on control/untreated units
            - threshold: the average of the outcome on treated units
            - nmatches: the average treatment effects between treated and control units
        The values of this list will be used to draw barplots in the web-application (front)
        """
        self.Treatment_W = Treatment_W
        self.Covariates_X = Covariates_X
        self.outcome_Y = outcome_Y
        
        self.Treatment_name = Treatment_W.name
        self.outcome_name = outcome_Y.name
        self.X_colnames = Covariates_X.columns.tolist()
        self.obs_df = pd.concat([Covariates_X, Treatment_W, outcome_Y], axis=1)
        
        """ 
        Advanced options
            - model: the average of the outcome on control/untreated units
            - threshold: the average of the outcome on treated units
            - nmatches: the average treatment effects between treated and control units
            - method
        The values of this list will be used to draw barplots in the web-application (front)
        """
        self.model = model
        self.method = method
        self.threshold = threshold
        self.max_rand = max_rand
        self.nmatches = nmatches

    def compute_apparentEffect(self) -> list:
        """ 
        Generate a list of three values (Avg_treated, Avg_ctrl and Observed_ATE):
            - Avg_ctrl: the average of the outcome on control/untreated units
            - Avg_treated: the average of the outcome on treated units
            - Observed_ATE: the average treatment effects between treated and control units
        The values of this list will be used to draw barplots in the web-application (front)
        """
        # Sample observed treated and control units    
        obs_treated = self.obs_df[self.obs_df[self.Treatment_name]==1]
        obs_ctrl = self.obs_df[self.obs_df[self.Treatment_name]==0]
         
        # Compute the average apparent effect before matching 
        Avg_treated = self.outcome_Y[self.Treatment_W == 1].mean()
        Avg_ctrl = self.outcome_Y[self.Treatment_W == 0].mean()
        Observed_ATE = Avg_treated - Avg_ctrl

        return [Avg_ctrl, Avg_treated, Observed_ATE]
    
    
    def tableone_before(self) -> pd.DataFrame:
        """ 
        Generate a DataFrame summarizing statistical information before matching and return
            - tableone_obs: the summary statistics table on both treated and control units
        The values of this DataFrame will be exported and shown in a pdf on the the web-application
        """
        table_data = self.obs_df
        table_data.reset_index(drop = True, inplace = True)
        tableone_obs = TableOne(table_data, columns = self.X_colnames, 
                                groupby = self.outcome_name, pval = True) 

        return tableone_obs

    
    def propensity_estimate(self) -> np.ndarray:
        """ 
        Given a chosen classification model, generate a vector  and return
            - ps_estimate: a vector containing the values of propensity score for each unit
        """
        if self.model == "LogisticRegression":
            ps_model = LogisticRegression().fit(self.Covariates_X, self.Treatment_W)
            ps_estimate =  ps_model.predict_proba(self.Covariates_X)[:, 1]
        elif self.model == "RandomForest":
            ps_model = RandomForestClassifier().fit(self.Covariates_X, self.Treatment_W)
            ps_estimate =  ps_model.predict_proba(self.Covariates_X)[:, 1]
        elif self.model == "GradientBoosting":
            ps_model = GradientBoostingClassifier().fit(self.Covariates_X, self.Treatment_W)
            ps_estimate = ps_model.predict_proba(self.Covariates_X)[:, 1]
        return ps_estimate
    
    
    def matching(self) -> list:
        """ 
        Run the algorithm of matching and generate two lists:
            - match_ids: a list containing the index of each treated unit
            - result: a list containing the index of the matched control units
        """
        # Estimate the propensity score
        ps_estimate = self.propensity_estimate()
        data_ps = self.obs_df.assign(propensity_score = ps_estimate)

        # Sample treated and control unit with their estimated propensity score
        treated_scores = data_ps[self.Treatment_W==1][['propensity_score']]
        n_treated = len(treated_scores)
        ctrl_scores = data_ps[self.Treatment_W==0][['propensity_score']]

        # Begin the algorithm of Matching
        match_ids, result = [], []
        for i in range(n_treated):
            score = treated_scores.iloc[i]
            if self.method == 'random':
                # Select randomly nmatch units such that the propensity score is below threshold
                bool_match = abs(ctrl_scores - score) <= self.threshold
                matches = ctrl_scores.loc[bool_match[bool_match.propensity_score].index]
            elif self.method == 'min':
                # Select the most closets nmatch units such that the propensity score is below threshold
                matches = abs(ctrl_scores - score).sort_values('propensity_score').head(self.nmatches)
            if len(matches) == 0:
                continue
            # Randomly choose nmatches indices, if len(matches) > nmatches
            select = self.nmatches if self.method != 'random' else np.random.choice(range(1, self.max_rand+1), 1)
            chosen = np.random.choice(matches.index, min(select, self.nmatches), replace = False)
            
            # Append the index of each unit and the corresponding matched treated units
            match_ids.extend([i] * (len(chosen)+1))
            result.extend([treated_scores.index[i]] + list(chosen))
            
        return match_ids, result
    
    
    def matching_df(self) -> pd.DataFrame:
        """ 
        Generate a pandas DataFrame with matched units:
            - matched_data: whose columns are the same as the observational dataset and two
            additional columns designing the indices of treated units and the corresponding matched
            control units 
        """        
        # Run the matching algorithm
        match_ids, result = self.matching()
        
        # Create the matching dataset of treated and control units
        matched_data = self.obs_df.loc[result]
        matched_data['match_id'] = match_ids # the id of matched unit for control units
        matched_data['record_id'] = matched_data.index # the id of each unit of treated units
    
        return matched_data
    
    
    def compute_causalEffect(self) -> list:
        """ 
        Generate a list of three values (matched_Avg_ctrl, matched_Avg_treated and causal_ATE):
            - matched_Avg_ctrl: the average of the outcome on control/untreated units
            - matched_Avg_treated: the average of the outcome on treated units
            - causal_ATE: the average treatment effects between treated and control units
        The values of this list will be used to draw barplots in the web-application (front)
        """
  
        # Create the matching dataset of treated and control units
        matched_data = self.matching_df()
        
        # Sample matched treated and control units    
        matched_treated = matched_data[matched_data[self.Treatment_name]==1]
        matched_ctrl = matched_data[matched_data[self.Treatment_name]==0]
    
        # Compute the average causal effect after matching  
        matched_Avg_treated = matched_treated[self.outcome_name].mean()
        matched_Avg_ctrl = matched_ctrl[self.outcome_name].mean()
        causal_ATE = matched_Avg_treated - matched_Avg_ctrl
    
        return [matched_Avg_ctrl, matched_Avg_treated, causal_ATE]
    
    def tableone_after(self) -> pd.DataFrame:
        """ 
        Generate a DataFrame summarizing statistical information after matching
            - tableone_obs: the summary statistics table on both treated and control units
        The values of this DataFrame will be exported and shown in a pdf on the the web-application
        """
        # Create the matching dataset of treated and control units
        matched_data = self.matching_df()
        
        matched_table_data = matched_data.drop(['match_id', 'propensity_score', 'record_id'], axis = 1)
        matched_table_data.reset_index(drop=True, inplace=True)
        matched_table_data.index = range(1,matched_table_data.shape[0]+1)
        tableone_matched = TableOne(matched_table_data, columns = self.X_colnames, 
                                groupby = self.outcome_name, pval = True) 

        return tableone_matched
    
    def get_densities_plots(self) -> dict:
        """ 
        Generate two dictionnary containing data points of density plots
            - data_points_before: a dictionary whose keys are the covariates and values are the data
            points of the corresponding density plot before matching for the treated/contro units and
            the mean value of the covariate.
            - data_points_after: Same structure as data_points_before but after matching units.
        The values of these dictionnary will be used to plot density figures and export them
        in the web-application (PDF).
        """
        # Sample treated and control units    
        obs_treated = self.obs_df[self.obs_df[self.Treatment_name]==1]
        obs_ctrl = self.obs_df[self.obs_df[self.Treatment_name]==0]
        
        # Sample matched treated and control units    
        matched_data = self.matching_df()
        matched_treated = matched_data[matched_data[self.Treatment_name]==1]
        matched_ctrl = matched_data[matched_data[self.Treatment_name]==0]
        
        ncol = self.Covariates_X.shape[0]
        data_points_before, data_points_after = {}, {}
        
        for j in range(ncol):
            Xj = self.X_colnames[j] 
            
            # Extract data points and the mean value before matching for treated units
            x_tr_before, y_tr_before = sns.kdeplot(obs_treated[Xj]).lines[0].get_data()
            mean_tr_before = obs_treated[Xj].mean()
            
            # Extract data points and the mean value before matching for control units
            x_ctrl_before, y_ctrl_before = sns.kdeplot(obs_ctrl[Xj]).lines[0].get_data()
            mean_ctrl_before = obs_ctrl[Xj].mean()

            # Stock these values in the dictionnary whose key value is the column name
            data_points_before[Xj] = [x_tr_before, y_tr_before, mean_tr_before, 
                                      x_ctrl_before, y_ctrl_before, mean_ctrl_before]        

            # Do the same for the matched dataset
            x_tr_after, y_tr_after = sns.kdeplot(matched_treated[Xj]).lines[0].get_data()
            mean_tr_after = matched_treated[Xj].mean()
            
            x_ctrl_after, y_ctrl_after = sns.kdeplot(matched_ctrl[Xj].drop_duplicates()).lines[0].get_data()
            mean_ctrl_after = matched_ctrl[Xj].drop_duplicates().mean()
            
            data_points_after[Xj] = [x_tr_after, y_tr_after, mean_tr_after, 
                                      x_ctrl_after, y_ctrl_after, mean_ctrl_after]     
            
        return data_points_before, data_points_after 



if __name__ == "__main__": 
    # Import the default dataset (cleaned churning datased)
    clean_df = pd.read_csv("BankChurners_cleaned.csv")
    
    # Set the treatment T, covariates X and the outcome Y, all advanced parameters to default values
    Treatment_W = clean_df['Card_Category']
    outcome_Y = clean_df['Attrition_Flag']
    Covariates_X = clean_df.drop(columns=[Treatment_W.name, outcome_Y.name])
    
    # Instantiate the Class PropensityMatching
    PSM_model = PropensityMatching(Treatment_W, Covariates_X, outcome_Y)
    
    # Compute the observed effect of having a VIP card
    observed_res = PropensityMatching.compute_apparentEffect(PSM_model)
    Observed_ATE = 100*observed_res[2]
    print("The Observed treatment effect of having a VIP Credit Card : " + str("{:.2f}".format(float(Observed_ATE))) + ' %')
    
    # Compute the causal effect of having a VIP card
    causal_res = PropensityMatching.compute_causalEffect(PSM_model)
    Causal_ATE = 100*causal_res[2]
    print("The causal treatment effect of having a VIP Credit Card : " + str("{:.2f}".format(float(Causal_ATE))) + ' %')
    