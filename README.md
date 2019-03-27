# WifiTrafficAnalyzer

Class to process and analyze real recorded wifi data or simulated data produced from provided transition matrix.

## arguments and inputs

### top level arguments:
- mode {str} -- ['sim', 'real']
    - 'sim': simulate data through provided transition matrix
    - 'real': process and analyze recorded wifi traffic
- path_dict {dict} -- dictionary of file paths (must be `pathlib.Path` objects)

### path_dict configs
There are two provided dictionaries of data paths, `real_data_path_dict` and `transition_matrics_path_dict`. Each dictionary contains a unique key used to select the desired file and a `pathlib.Path` object referencing the desired file. All files must be valid in order to execute, remove invalid or unnecessary paths by commenting out the lines.

```python
real_data_path_dict = {
    'real_2': Path(r'data/wifitrafficstats2.csv'),
    'real_3': Path(r'data/wifitrafficstats3.csv'),
    'real_4': Path(r'data/wifitrafficstats4.csv'),
    #'real_5': Path(r'data/wifitrafficstats5.csv'),    <--- inactivated path
    'real_6': Path(r'data/wifitrafficstats6.csv')
}
```

### chart formatting arguments
There are four arguments to the `.state_length_vs_timestep_*()` functions for real and sim data:
- chart_type {str} -- ['bar', 'area']
    - 'bar': classic histogram bar chart
    - 'area': area chart (filled line chart)
- background_color {str} -- hex code for chart background color, ex:
    - '#abb2bf' - grey
    - '#ffffff' - white
- lower_bound {int} -- lower bound for duration (x-axis)
- upper_bound {int} -- upper bound for duration (x-axis)

## execution walkthrough

### real data execution
Instantiate the class in `real` mode:
```python
WTA = WifiTrafficAnalyzer(mode='real', path_dict=real_data_path_dict)
```

The class will respond with all valid paths for selection:
```python
initializing..
initialization complete, mode: real
real data options:
	key: real_2, path: data\wifitrafficstats2.csv
	key: real_3, path: data\wifitrafficstats3.csv
	key: real_4, path: data\wifitrafficstats4.csv
	key: real_5, path: data\wifitrafficstats5.csv
	key: real_6, path: data\wifitrafficstats6.csv
           ^-- `tar_key`
```

Select and process the desired file by calling the `.process_real_data()` method with the desired file key:
```python
WTA.process_real_data('real_4')
```

To view the state length distribution plots, call the `.state_length_vs_timestep_real()` method with desired formatting arguments:
```python
WTA.state_length_vs_timestep_real(
    chart_type='bar',
    background_color='#abb2bf',
    lower_bound=25,
    upper_bound=2500
)
```


### sim data execution
Instantiate the class in `sim` mode:
```python
WTA = WifiTrafficAnalyzer(mode='sim', path_dict=transition_matrices_path_dict)
```

The class will respond with all valid paths for selection:
```python
initializing..
initialization complete, mode: sim
sim data options:
	key: tmat_1, path: data\wifi_t_matrices.csv
	key: tmat_2, path: data\wifi_t_matrices2.csv
           ^-- `tar_key`
```

Select and process the desired file by calling the `.process_transition_matrices()` method with the desired file key:
```python
WTA.process_transition_matrices('tmat_2')
```

Simulate data for all transition matrices in the selected file by calling the `.simulate_all_transition_matrices()` method and changing the desired `n_samples` and `m_trials` arguments as required:
```python
WTA.simulate_all_transition_matrices(n_samples=100_000, m_trials=10)
```

To view the interactive state length distribution plots, call the `.state_length_vs_timestep_sim()` method with desired formatting arguments:
```python
WTA.state_length_vs_timestep_sim(
    chart_type='bar',
    background_color='#abb2bf',
    lower_bound=25,
    upper_bound=200
)
```

### code only example

```python
WTA_real = WifiTrafficAnalyzer(mode='sim', path_dict=transition_matrices_path_dict)
WTA_real.process_transition_matrices('tmat_2')
WTA_real.simulate_all_transition_matrices(n_samples=1_000_000, m_trials=100)
WTA_real.state_length_vs_timestep_sim(
    chart_type='bar',
    background_color='#abb2bf',
    lower_bound=25,
    upper_bound=200
)
```
