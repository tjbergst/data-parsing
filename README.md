# `DataParserLVM`
(but not really LVM, you know?)

## arguments and inputs

### top level arguments:
The input arguments to class `DataParserLVM` are as follows
- `lvm_path` : str
    - `pathlib.Path` object pointing to recorded data file
    - can be set in `params` section of notebook
- `source` : str -- [`txt`, `lvm`]
    - determines parsing method for input file
    - `txt`: may be a `.txt` or `.lvm` file, however, it cannot have headers and must be in tab delimited format
        - this is for data files just like `test_packet.lvm`
        - will call `parse_other_file()` method
    - `lvm`: must be `.lvm` file, must include header or will not parse correctly
        - this is for data files just like `qim_20kbps_10db_l2_v2.lvm`
        - will call `parse_lvm_file()` method followed by `read_csv_data()` method (`.csv` data is written to disk as `f'{file_name}_parsed.csv')` where `file_name` is the original file name
- `header` : `None` (preferred) or `list` or `np.ndarray`
    - aka preamble
    - if `None`, automatically generates header, else use provided
    - automatically generated header is configured for `spb=12`
- `preprocess_mode`: str -- [`static`, `dynamic`]
    - determines how to set `th` for conversion to `1`'s and `0`'s
    - `static`: compute `th` as `1/2 * max(input_data)`
    - `dynamic`: compute `th` as `min(input_data) + 1/2 * (max(input_data) - min(input_data))`

### other notable attributes
- `self.data_msg`: the known transmitted data message
    - this has been tuned to `spb=12` and is constructed in the `create_data_message()` method
    - the base array is `1,0` repeated 10 times, this array is then repeated array-wise twice to create your `b0`, the base array is then repeated twice element-wise to create your `b1`. This set of `b0` + `b1` is then repeated once more array-wise to create the full `data_msg` consisting of 160 bits. See the noted method for the full definition.
- `self.data`: the parsed and processed input data (non-discretized)
- `self.discretized_array`: the discretized version of the input data
- `self.ber`: a list of the computed bit error rates


## user called methods

### `discretize_signal(spb)` method
This method takes the parsed and preprocessed input data and converts it to a discrete signal using the provided samples per bit `spb`. Analysis of the `test_packet.lvm` file indicates optimum results with `spb=12` -- the `header` and `data_msg` have been tuned to this. If an `spb` other than `12` is used, these variables will need to be adjusted.

### `compute_bit_error_rate()` method
This method computes the bit error rate of the discretized signal. First, the starting indices of all known headers are computed and the number of matches are printed. The start and stop parameters are defined as follows:
- `start`: starting index of header + `length(header)` + 4 (4 `0`'s always follow header)
- `stop`: `start` + `length(data_msg)`
The discretized data is then sliced and an element-wise difference between the sliced data and the known `self.data_msg` is computed. The difference array is then summed and stored in the list `self.ber`

## utility methods

### `get_signal_stats(data)` method
This method returns frequency counts of the state lengths of the provided `data` in order of occurrence. Sample output (format is (`state_length`, `frequency`)):
```python
DP.get_signal_stats(DP.data)
>>> [(14, 39780),
    (10, 39722),
    (22, 19880),
    (26, 19877),
    (11, 14791),
    (13, 14733),
    (25, 7383),
    (23, 7380),
    (599, 1273),
    (49, 1190),
    (48, 173),
    (600, 90),
    (5, 1)]
```

### `plot_subset(data, start, stop)` method
This method can be used to plot subsets of the provided data. `start` and `stop` are indices. If called after computing the bit error rate, the header starts can be used to examine individual messages, example:
```python
DP.plot_subset(
    DP.discretized_array,          # <-- discrete signal as input data
    start=DP.header_starts[0],     
    stop=DP.header_starts[0] + len(DP.header) + 4 + len(DP.data_msg)
)
```
Here we use the first located header as the `start` slice, and the same `start` plus the length of the remaining message as described in `compute_bit_error_rate` section. By changing the `idx` of `DP.header_starts[idx]` you can view any message. Note that both `DP.discretized_array` and `DP.data` can be used as input.

### `output_discretized_array(output_path, output_format)` method
This method can be used to output the discretized array to a `.csv` file. An `output_path='auto'` will save the data to `data/discretized_array.csv`. Other locations may be specified as `pathlib.Path` objects (preferred) or strings.


## usage examples

### read and process tab delimited (no header) dataset

```python
lvm_path = Path(r'data/test_packet.lvm')

DP = DataParserLVM(
    lvm_path,
    source='txt',
    preprocess_mode='dynamic'
)

DP.discretize_signal(spb=12)
DP.compute_bit_error_rate()
```

### read and process lvm dataset

```python
lvm_path = Path(r'data/qim_20kbps_10db_l2_v2.lvm')

DP = DataParserLVM(
    lvm_path,
    source='lvm',
    preprocess_mode='static'
)

DP.discretize_signal(spb=12)
DP.compute_bit_error_rate()
```
