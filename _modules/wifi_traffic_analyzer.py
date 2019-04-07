import pandas as pd
import numpy as np
from tqdm import tqdm_notebook, tnrange, trange, tqdm
from itertools import cycle
from collections import Counter
from pathlib import Path
import time

import altair as alt
from altair.expr import datum




class WifiTrafficAnalyzer:
    def __init__(self, mode, path_dict): 
        print('initializing WTA..')
        
        # set mode
        mode_options = ['sim','real']
        self.mode = mode.strip().lower()
        assert self.mode in mode_options, \
            f'error: invalid mode {self.mode}, options are {mode_options}'
        
        # load path dict
        self.path_dict = path_dict
        for path in self.path_dict.values():
            try:
                assert path.exists()
            except AssertionError:
                print(f'no file found at path: {path}')
            except AttributeError:
                print('invalid path format, must be pathlib.Path object')
        
        print(f'initialization complete, mode: {self.mode}')
        print(f'{self.mode} data options: ')
        for key, path in self.path_dict.items():
                print(f'\tkey: {key}, path: {path}')
                
        self.sim_data_output_path = lambda tstep: Path(f'data/sim_data_{tstep}_microsec.csv')
        
        
    
    #------------------------------ real data ------------------------------
    def process_real_data(self, tar_key):
        '''
        takes csv of recorded wifi traffic (identified by key) and stores 
        a dataframe ready to be turned into data list ([1,1,1,0,0,1,1,0,0,0])
        '''
        # validate target file selection
        try:
            assert tar_key in self.path_dict.keys()
        except AssertionError:
            print(f'invalid key {tar_key}, options are {list(self.path_dict.keys())}')
            
        real_df = (
            pd
            .read_csv(self.path_dict[tar_key])
            .rename({ 
                'Time':'time', 
                'Length':'length', 
                'TX Rate':'rate'
            }, axis=1)
            .eval('time = time * 1e6')
            .eval('rate = rate * 1e6')
            .eval('length = length * 8')
            .eval('on_time = 1e6 * length / rate')
            .eval('end_time = time + on_time')
            .assign(
                shift_time = lambda x: x.end_time.shift(1),
                off_time = lambda x: [
                    (time - shift_time) 
                        if (time - shift_time) > 0 else 0 
                    for time, shift_time in (zip(x.time, x.shift_time))
            ])
            .drop(['No.','Delta Time','shift_time'], axis=1)
            .round(0)
        )
        
        self.generate_real_data_list(real_df)
    
    
    def generate_real_data_list(self, df):
        '''
        takes dataframe with 'on_time' and 'off_time' time stamp columns 
        and stores a numpy array of 1's (on_time) and 0's (off_time)
        '''
        
        real_data_list = []
    
        #for row in tqdm_notebook(df[['on_time','off_time']].iloc[1:].itertuples(), total=df.shape[0]-1):
        for row in df[['on_time','off_time']].iloc[1:].itertuples():
            real_data_list.extend([1 for i in range(int(row.on_time))])
            real_data_list.extend([0 for i in range(int(row.off_time))])
        
        self.real_data_array = np.array(real_data_list, copy=True)
        
        self.format_real_data()
        
    def format_real_data(self):
        data_0, data_1 = self.get_split_state_lengths(self.real_data_array)
        
        df_0 = (
            pd
            .DataFrame(
                Counter(data_0).most_common(), 
                columns=['duration','frequency']
            )
            .sort_values('duration')
            .reset_index(drop=True)
            .assign(bit = 0)
        )
        
        df_1 = (
            pd
            .DataFrame(
                Counter(data_1).most_common(), 
                columns=['duration','frequency']
            )
            .sort_values('duration')
            .reset_index(drop=True)
            .assign(bit = 1)
        )
        
        self.full_real_df = pd.concat([df_0, df_1])
        
    
    #------------------------------ sim data ------------------------------
    def process_transition_matrices(self, tar_key):
        # validate target file selection
        try:
            assert tar_key in self.path_dict.keys()
        except AssertionError:
            print(f'invalid key {tar_key}, options are {list(self.path_dict.keys())}')
        
        
        self.tmat_df = (
            pd
            .read_csv(
                self.path_dict[tar_key], 
                names=['OnOn','OnOff','OffOn','OffOff','timestep']
            )
            #.eval('OffOff = 1 - OffOff')
            #.eval('OffOn = 1 - OffOn')
            [['OnOn', 'OnOff', 'OffOff', 'OffOn', 'timestep']]
        )
        
    
    
    def generate_sim_data(self, n_samples, m_trials, transition_matrix):
        '''
        takes a transition matrix and generates a simulated signal
        with <n_samples> bits and <m_trials> trials, stores an n x m 
        matrix
        '''
            
        cycle_dict = {
            (1,1,1): 0,
            (1,1,0): 1,
            (1,0,1): 2,
            (1,0,0): 1,
            (0,1,1): 1,
            (0,1,0): 2,
            (0,0,1): 1,
            (0,0,0): 0,
        }
        
        global p
        p = 0.5
        sim_matrix = np.zeros((int(m_trials), int(n_samples)))        
        state_cycler = cycle(transition_matrix)
        
        def cycle_states():
            return next(state_cycler)
        
        def generate_bit(n):
            global p
            
            if n == 0:
                return np.random.binomial(1, p)
                
            else:
                for i in range(n):
                    p = cycle_states()
    
                return np.random.binomial(1, p)
            
        for i in range(m_trials):
            sim_data = [1,1,1]
            p = 0.5
            
            for _ in range(n_samples + 1):        
                new_bit = generate_bit(cycle_dict[tuple(sim_data[-3:])])        
                sim_data.append(new_bit)
                
            sim_matrix[i,:] = np.array(sim_data[4:])
        
        return sim_matrix
    
    
    def simulate_all_OP_transition_matrices(self, tmat_dataframe, n_samples=int(1e5), m_trials=10):        
        start_time = time.time()
        sim_data_matrix = np.zeros((tmat_dataframe.shape[0] + 1, n_samples))        
        
        #for row in tqdm_notebook(tmat_dataframe.itertuples(), total=tmat_dataframe.shape[0]):
        for row in tmat_dataframe.itertuples():
            transition_matrix = [row.OnOn, row.OnOff, row.OffOff, row.OffOn]
            sim_data_matrix[row.Index + 1, :] = self.generate_sim_data(n_samples, m_trials, transition_matrix)            
            
        print(f'total elapsed time: {(time.time() - start_time)/60:0.2f} minutes')
        
        return sim_data_matrix.astype('int')
    
        
    def simulate_all_transition_matrices(self, tmat_dataframe=None, n_samples=int(1e5), m_trials=10, 
                                               postprocess=True, output_data=False):        
        if tmat_dataframe is None:
            tmat_dataframe = self.tmat_df
        
        start_time = time.time()

        self.stats_dict = self.master_dictionary = {
            row.timestep: {}
            for row in tmat_dataframe.itertuples()
        }
        
        for row in tqdm_notebook(tmat_dataframe.itertuples(), total=tmat_dataframe.shape[0]):
            transition_matrix = [row.OnOn, row.OnOff, row.OffOff, row.OffOn]
            sim_matrix = self.generate_sim_data(n_samples, m_trials, transition_matrix)    
            
            self.master_dictionary[row.timestep] = self.compute_sim_stats(sim_matrix)
            
            # output data
            if output_data:
                np.savetxt(
                    self.sim_data_output_path(row.timestep),
                    sim_matrix,
                    delimiter=','
                )

            
        print(f'total elapsed time: {(time.time() - start_time)/60:0.2f} minutes')
        
        if postprocess:
            self.post_process_sim_stats()
    
    
    
    def compute_sim_stats(self, sim_matrix):
        sample_size = sim_matrix.shape[1]
        stats = sim_matrix.sum(axis=1)
                        
        state_lengths_0 = [
            item for sublist in
            [
                list(self.get_split_state_lengths(sim_matrix[i,:])[0])
                for i in range(sim_matrix.shape[0])
            ]
            for item in sublist
        ]
        
        state_lengths_1 = [
            item for sublist in
            [
                list(self.get_split_state_lengths(sim_matrix[i,:])[1])
                for i in range(sim_matrix.shape[0])
            ]
            for item in sublist
        ]
        
        return {
            'mean': np.mean(stats) / sample_size,
            'std': np.std(stats) / sample_size,
            'state_lengths_0': Counter(state_lengths_0).most_common(),
            'state_lengths_1': Counter(state_lengths_1).most_common()
        }
    
    def post_process_sim_stats(self):
        '''
        takes previously computed self.master_dictionary and splits it into two dataframes
            full_df: duration, frequency, bit, timestep for all simulations
            stats_df: mean and std of m_trials
        '''
        
        self.df_dict_0 = {
            tstep: (
                pd.DataFrame(
                    stats['state_lengths_0'], 
                    columns=['duration','frequency']
                )
                .sort_values('duration')
                .reset_index(drop=True)
                .assign(
                    timestep = tstep,
                    bit = 0
                )
            )
            for tstep, stats in self.master_dictionary.items()
        }
        
        self.df_dict_1 = {
            tstep: (
                pd.DataFrame(
                    stats['state_lengths_1'], 
                    columns=['duration','frequency']
                )
                .sort_values('duration')
                .reset_index(drop=True)
                .assign(
                    timestep = tstep,
                    bit = 1
                )
            )
            for tstep, stats in self.master_dictionary.items()
        }
        
        self.full_sim_df = pd.concat(
            [df for df in self.df_dict_0.values()] + [df for df in self.df_dict_1.values()]
        )
        
        stats_list = [
            {
                'timestep': timestep,
                'mean': stats['mean'],
                'std': stats['std']
            }
            for timestep, stats in self.master_dictionary.items()
        ]
        
        self.stats_df = pd.DataFrame(stats_list)[['timestep','mean','std']]
    
    
    
    #------------------------------ common functions ------------------------------
    def get_split_state_lengths(self, data):
        '''
        takes data list ([1,1,0,1,0,0,...]) and returns separate
        numpy arrays of duration of consecutive bits ([13,245,2588,19,1056,...])
        for 1's and for 0's
        '''
        
        data = np.array(data)    
        
        state_lengths = np.diff(
            np.where(
                np.concatenate(
                    ([data[0]],
                     data[:-1] != data[1:],
                     [0]
                    )
                )
            )[0]
        )[::2]
        
        # split
        if data[0] == 0:
            data_0 = state_lengths[0::2]
            data_1 = state_lengths[1::2]
        else:
            data_0 = state_lengths[1::2]
            data_1 = state_lengths[0::2]
            
        return data_0, data_1
    
    
   
    
    #------------------------------ visualization ------------------------------
    def state_length_vs_timestep_real(self, chart_type='bar', background_color='#abb2bf', lower_bound=25, upper_bound=2500):
        '''
        creates interactive charts showing distributions of on and off times for simulated data
        
        chart types:
            bar: bar chart (histogram)
            area: area chart (filled line chart)
            
        background_color:
            hex code for chart background color, ex:
                #abb2bf - grey
                #ffffff - white
        '''
        
        full_real_df = (
            self
            .full_real_df
            .groupby(['bit','duration'])
            .frequency
            .sum()
            .reset_index()
        )
       
        
        detail_bar_chart = alt.Chart(
            full_real_df,
            height=375,
            width=800
        ).mark_bar(
            opacity=0.5
        ).transform_filter(
            datum.duration > lower_bound
        ).transform_filter(
            datum.duration < upper_bound
        ).encode(
            alt.X('duration:Q'),
            alt.Y(
                'frequency:Q', 
                #scale=alt.Scale(type='log')
            ),
            tooltip=[
                alt.Tooltip('duration:Q'),
                alt.Tooltip('frequency:Q'),
                alt.Tooltip('bit:N'),
            ]
        )
        
        detail_area_chart = alt.Chart(
            full_real_df,
            height=375,
            width=800
        ).mark_area(
            opacity=0.5
        ).transform_filter(
            datum.duration > lower_bound
        ).transform_filter(
            datum.duration < upper_bound
        ).encode(
            alt.X('duration:Q'),
            alt.Y(
                'frequency:Q', 
                #scale=alt.Scale(type='log')
            ),
            tooltip=[
                alt.Tooltip('duration:Q'),
                alt.Tooltip('frequency:Q'),
                alt.Tooltip('bit:N'),
            ]
        )
        
        if chart_type == 'bar':
            detail_chart = detail_bar_chart
        elif chart_type == 'area':
            detail_chart = detail_area_chart
        else:
            print(f'unsupported chart type {chart_type}')
        
        
        stacked_bit_details = alt.vconcat(
            detail_chart.transform_filter(datum.bit == 0).properties(title='off time distributions'),
            detail_chart.transform_filter(datum.bit == 1).properties(title='on time distributions'),
            background=background_color
        )
        
        display(stacked_bit_details)       
        
    
    
    
    def state_length_vs_timestep_sim(self, chart_type='bar', background_color='#abb2bf', lower_bound=25, upper_bound=200):
        '''
        creates interactive charts showing distributions of on and off times for simulated data
        
        chart types:
            bar: bar chart (histogram)
            area: area chart (filled line chart)
            
        background_color:
            hex code for chart background color, ex:
                #abb2bf - grey
                #ffffff - white
        '''
        
        full_sim_df = self.full_sim_df
        chart_type = chart_type.strip().lower()

        sel_timestep = alt.selection_multi(encodings=['y'])
        
        bar_chart = alt.Chart(
            full_sim_df,
            height=800,
            width=250
        ).mark_bar(
        ).encode(
            alt.X(
                'max(duration):Q',
                title='max packet duration',
                scale=alt.Scale(type='log')
            ),
            alt.Y(
                'timestep:N',
            ),
            color=alt.condition(
                sel_timestep,
                'timestep:N',
                alt.value('#96989b'),
                legend=None
            ),
            tooltip = [
                alt.Tooltip('timestep:N'),
                alt.Tooltip('duration:Q', aggregate='max')
            ]
        ).add_selection(
            sel_timestep
        )  
        
        
        #---------- histograms ----------
        detail_bar_chart = alt.Chart(
            full_sim_df,
            height=375,
            width=800
        ).mark_bar(
            opacity=0.5
        ).transform_filter(
            sel_timestep
        ).transform_filter(
            datum.duration > lower_bound
        ).transform_filter(
            datum.duration < upper_bound
        ).encode(
            alt.X('duration:Q'),
            alt.Y(
                'frequency:Q', 
                #scale=alt.Scale(type='log')
            ),
            color=alt.Color('timestep:N', legend=None),
            tooltip=[
                alt.Tooltip('duration:Q'),
                alt.Tooltip('frequency:Q'),
                alt.Tooltip('timestep:N'),
                alt.Tooltip('bit:N'),
            ]
        )
        
        detail_area_chart = alt.Chart(
            full_sim_df,
            height=375,
            width=800
        ).mark_area(
            opacity=0.5
        ).transform_filter(
            sel_timestep
        ).transform_filter(
            datum.duration > lower_bound
        ).transform_filter(
            datum.duration < upper_bound
        ).encode(
            alt.X('duration:Q'),
            alt.Y(
                'frequency:Q', 
                #scale=alt.Scale(type='log')
            ),
            color=alt.Color('timestep:N', legend=None),
            tooltip=[
                alt.Tooltip('duration:Q'),
                alt.Tooltip('frequency:Q'),
                alt.Tooltip('timestep:N'),
                alt.Tooltip('bit:N'),
            ]
        )
        
        if chart_type == 'bar':
            detail_chart = detail_bar_chart
        elif chart_type == 'area':
            detail_chart = detail_area_chart
        else:
            print(f'unsupported chart type {chart_type}')
        
        stacked_bit_details = alt.vconcat(
            detail_chart.transform_filter(datum.bit == 0).properties(title='off time distributions'),
            detail_chart.transform_filter(datum.bit == 1).properties(title='on time distributions'),
        )
        
        full = alt.hconcat(
            bar_chart,
            stacked_bit_details,    
            background=background_color
        )
        
        display(full)
    
        
    
    
    
    
        
        
        