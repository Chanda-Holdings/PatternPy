import pandas as pd
import numpy as np


# The detectors below compare numpy arrays instead of pandas Series: each
# pandas shift, compare and .loc assignment costs tens of microseconds.
# Output columns, values and dtypes are unchanged.

def _values(df, column):
    return df[column].to_numpy(dtype=float)


def _shift(values, periods):
    """Series.shift(periods) on an array: NaN where no bar exists."""
    out = np.full(len(values), np.nan)
    if periods > 0:
        out[periods:] = values[:-periods]
    elif periods < 0:
        out[:periods] = values[-periods:]
    else:
        out[:] = values
    return out


def _labels(size, *masks, empty=np.nan):
    """An object column: each later (mask, label) pair overwrites earlier ones."""
    column = np.full(size, empty, dtype=object)
    for mask, label in masks:
        column[mask] = label
    return column


def _trend(values, window):
    """Sign of last minus first value in each rolling window, as
    rolling(window).apply did: NaN when the window holds a NaN or an
    infinity, which pandas rolling treats as missing."""
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    change = values[window - 1:] - values[:len(values) - window + 1]
    trend = np.where(change > 0, 1.0, np.where(change < 0, -1.0, 0.0))
    windows = np.lib.stride_tricks.sliding_window_view(~np.isfinite(values), window)
    trend[windows.any(axis=1)] = np.nan
    out[window - 1:] = trend
    return out


def detect_head_shoulder(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    high, low = _values(df, 'High'), _values(df, 'Low')
    high_max, low_min = df['high_roll_max'].to_numpy(), df['low_roll_min'].to_numpy()
    high_prev, high_next = _shift(high, 1), _shift(high, -1)
    low_prev, low_next = _shift(low, 1), _shift(low, -1)
    # Create a boolean mask for Head and Shoulder pattern
    mask_head_shoulder = (high_max > high_prev) & (high_max > high_next) & (high < high_prev) & (high < high_next)
    # Create a boolean mask for Inverse Head and Shoulder pattern
    mask_inv_head_shoulder = (low_min < low_prev) & (low_min < low_next) & (low > low_prev) & (low > low_next)
    # Create a new column for Head and Shoulder and its inverse pattern and populate it using the boolean masks
    df['head_shoulder_pattern'] = _labels(
        len(df),
        (mask_head_shoulder, 'Head and Shoulder'),
        (mask_inv_head_shoulder, 'Inverse Head and Shoulder'),
    )
    return df

def detect_multiple_tops_bottoms(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()
    high_prev = _shift(_values(df, 'High'), 1)
    low_prev = _shift(_values(df, 'Low'), 1)
    close_prev = _shift(_values(df, 'Close'), 1)
    # Create a boolean mask for multiple top pattern
    mask_top = (df['high_roll_max'].to_numpy() >= high_prev) & (df['close_roll_max'].to_numpy() < close_prev)
    # Create a boolean mask for multiple bottom pattern
    mask_bottom = (df['low_roll_min'].to_numpy() <= low_prev) & (df['close_roll_min'].to_numpy() > close_prev)
    # Create a new column for multiple top bottom pattern and populate it using the boolean masks
    df['multiple_top_bottom_pattern'] = _labels(
        len(df), (mask_top, 'Multiple Top'), (mask_bottom, 'Multiple Bottom')
    )
    return df

def calculate_support_resistance(df, window=3):
# Define the rolling window
    roll_window = window
    # Set the number of standard deviation
    std_dev = 2
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    # Calculate the mean and standard deviation for High and Low
    mean_high = df['High'].rolling(window=roll_window).mean()
    std_high = df['High'].rolling(window=roll_window).std()
    mean_low = df['Low'].rolling(window=roll_window).mean()
    std_low = df['Low'].rolling(window=roll_window).std()
    # Create a new column for support and resistance
    df['support'] = mean_low - std_dev * std_low
    df['resistance'] = mean_high + std_dev * std_high
    return df

def detect_triangle_pattern(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    high_max, low_min = df['high_roll_max'].to_numpy(), df['low_roll_min'].to_numpy()
    high_prev = _shift(_values(df, 'High'), 1)
    low_prev = _shift(_values(df, 'Low'), 1)
    close = _values(df, 'Close')
    close_prev = _shift(close, 1)
    # Create a boolean mask for ascending triangle pattern
    mask_asc = (high_max >= high_prev) & (low_min <= low_prev) & (close > close_prev)
    # Create a boolean mask for descending triangle pattern
    mask_desc = (high_max <= high_prev) & (low_min >= low_prev) & (close < close_prev)
    # Create a new column for triangle pattern and populate it using the boolean masks
    df['triangle_pattern'] = _labels(
        len(df), (mask_asc, 'Ascending Triangle'), (mask_desc, 'Descending Triangle')
    )
    return df

def detect_wedge(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    high, low = _values(df, 'High'), _values(df, 'Low')
    df['trend_high'] = _trend(high, roll_window)
    df['trend_low'] = _trend(low, roll_window)
    high_max, low_min = df['high_roll_max'].to_numpy(), df['low_roll_min'].to_numpy()
    trend_high, trend_low = df['trend_high'].to_numpy(), df['trend_low'].to_numpy()
    high_prev, low_prev = _shift(high, 1), _shift(low, 1)
    # Create a boolean mask for Wedge Up pattern
    mask_wedge_up = (high_max >= high_prev) & (low_min <= low_prev) & (trend_high == 1) & (trend_low == 1)
    # Create a boolean mask for Wedge Down pattern
    mask_wedge_down = (high_max <= high_prev) & (low_min >= low_prev) & (trend_high == -1) & (trend_low == -1)
    # Create a new column for Wedge Up and Wedge Down pattern and populate it using the boolean masks
    df['wedge_pattern'] = _labels(
        len(df), (mask_wedge_up, 'Wedge Up'), (mask_wedge_down, 'Wedge Down')
    )
    return df

def detect_channel(df, window=3):
    # Define the rolling window
    roll_window = window
    # Define a factor to check for the range of channel
    channel_range = 0.1
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    high, low = _values(df, 'High'), _values(df, 'Low')
    df['trend_high'] = _trend(high, roll_window)
    df['trend_low'] = _trend(low, roll_window)
    high_max, low_min = df['high_roll_max'].to_numpy(), df['low_roll_min'].to_numpy()
    trend_high, trend_low = df['trend_high'].to_numpy(), df['trend_low'].to_numpy()
    high_prev, low_prev = _shift(high, 1), _shift(low, 1)
    narrow = high_max - low_min <= channel_range * (high_max + low_min)/2
    # Create a boolean mask for Channel Up pattern
    mask_channel_up = (high_max >= high_prev) & (low_min <= low_prev) & narrow & (trend_high == 1) & (trend_low == 1)
    # Create a boolean mask for Channel Down pattern
    mask_channel_down = (high_max <= high_prev) & (low_min >= low_prev) & narrow & (trend_high == -1) & (trend_low == -1)
    # Create a new column for Channel Up and Channel Down pattern and populate it using the boolean masks
    df['channel_pattern'] = _labels(
        len(df), (mask_channel_up, 'Channel Up'), (mask_channel_down, 'Channel Down')
    )
    return df

def detect_double_top_bottom(df, window=3, threshold=0.05):
    # Define the rolling window
    roll_window = window
    # Define a threshold to check for the range of pattern
    range_threshold = threshold

    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    high, low = _values(df, 'High'), _values(df, 'Low')
    high_max, low_min = df['high_roll_max'].to_numpy(), df['low_roll_min'].to_numpy()
    high_prev, high_next = _shift(high, 1), _shift(high, -1)
    low_prev, low_next = _shift(low, 1), _shift(low, -1)
    narrow_prev = (high_prev - low_prev) <= range_threshold * (high_prev + low_prev)/2
    narrow_next = (high_next - low_next) <= range_threshold * (high_next + low_next)/2

    # Create a boolean mask for Double Top pattern
    mask_double_top = (high_max >= high_prev) & (high_max >= high_next) & (high < high_prev) & (high < high_next) & narrow_prev & narrow_next
    # Create a boolean mask for Double Bottom pattern
    mask_double_bottom = (low_min <= low_prev) & (low_min <= low_next) & (low > low_prev) & (low > low_next) & narrow_prev & narrow_next

    # Create a new column for Double Top and Double Bottom pattern and populate it using the boolean masks
    df['double_pattern'] = _labels(
        len(df), (mask_double_top, 'Double Top'), (mask_double_bottom, 'Double Bottom')
    )

    return df

def detect_trendline(df, window=2):
      """
      Fully vectorized version using NumPy sliding windows (no loops).
      """
      n = len(df)

      if n < window + 1:
          df['slope'] = np.nan
          df['intercept'] = np.nan
          df['support'] = np.nan
          df['resistance'] = np.nan
          return df

      close_values = df['Close'].values

      # Create sliding windows view (no copying)
      from numpy.lib.stride_tricks import sliding_window_view
      windows = sliding_window_view(close_values, window)

      # Precompute x values
      x = np.arange(window)
      x_mean = x.mean()
      x_centered = x - x_mean
      x_denominator = np.sum(x_centered ** 2)

      # Vectorized computation across all windows
      y_means = windows.mean(axis=1)
      y_centered = windows - y_means[:, np.newaxis]

      # Calculate slopes and intercepts for all windows at once
      numerators = np.sum(x_centered * y_centered, axis=1)
      slopes_computed = numerators / x_denominator
      intercepts_computed = y_means - slopes_computed * x_mean

      # Pad with NaN for first 'window-1' rows (sliding window creates n-window+1 outputs)
      slopes = np.concatenate([np.full(window - 1, np.nan), slopes_computed])
      intercepts = np.concatenate([np.full(window - 1, np.nan), intercepts_computed])

      df['slope'] = slopes
      df['intercept'] = intercepts

      # Vectorized support/resistance calculation
      mask_support = slopes > 0
      mask_resistance = slopes < 0

      support_values = np.where(mask_support, close_values * slopes + intercepts, np.nan)
      resistance_values = np.where(mask_resistance, close_values * slopes + intercepts, np.nan)

      df['support'] = support_values
      df['resistance'] = resistance_values

      return df

def find_pivots(df):
    # Calculate differences between consecutive highs and lows
    high = _values(df, 'High')
    low = _values(df, 'Low')
    high_diffs = high - _shift(high, 1)
    low_diffs = low - _shift(low, 1)
    high_diffs_next = _shift(high_diffs, -1)
    low_diffs_next = _shift(low_diffs, -1)

    # Find higher high
    higher_high_mask = (high_diffs > 0) & (high_diffs_next < 0)

    # Find lower low
    lower_low_mask = (low_diffs < 0) & (low_diffs_next > 0)

    # Find lower high
    lower_high_mask = (high_diffs < 0) & (high_diffs_next > 0)

    # Find higher low
    higher_low_mask = (low_diffs > 0) & (low_diffs_next < 0)

    # Create signals column
    df['signal'] = _labels(
        len(df),
        (higher_high_mask, 'HH'),
        (lower_low_mask, 'LL'),
        (lower_high_mask, 'LH'),
        (higher_low_mask, 'HL'),
        empty='',
    )

    return df
