#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Script to power cycle DUT with external relay and measure current
#

from joulescope import scan_require_one
from joulescope.stream_buffer import stats_to_api
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib import ticker
import time
import re
import argparse
from colorama import Fore

def print_stats(data, sampling_frequency):
    is_finite = np.isfinite(data[:, 0])
    duration = len(data) / sampling_frequency
    finite = np.count_nonzero(is_finite)
    total = len(data)
    nonfinite = total - finite
    print(f'found {nonfinite} NaN out of {total} samples ({duration:.3f} seconds)')
    is_finite[0], is_finite[-1] = True, True  # force counting at start and end
    nan_edges = np.nonzero(np.diff(is_finite))[0]
    nan_runs = len(nan_edges) // 2
    if nan_runs:
        print(f'nan edges: {nan_edges.reshape((-1, 2))}')
        nan_edge_lengths = nan_edges[1::2] - nan_edges[0::2]
        run_mean = np.mean(nan_edge_lengths)
        run_std = np.std(nan_edge_lengths)
        run_min = np.min(nan_edge_lengths)
        run_max = np.max(nan_edge_lengths)
        print(f'found {nan_runs} NaN runs: {run_mean} mean, {run_std} std, {run_min} min, {run_max} max')
        
def plot_axis(axis, x, y, label=None):
    if label is not None:
        axis.set_ylabel(label)
    axis.grid(True)

    axis.plot(x, y)

    # draw vertical lines at start/end of NaN region
    yvalid = np.isfinite(y)
    for line in x[np.nonzero(np.diff(yvalid))]:
        axis.axvline(line, color='red')

    # Fill each NaN region, too
    ymin, ymax = np.min(y[yvalid]), np.max(y[yvalid])
    collection = matplotlib.collections.BrokenBarHCollection.span_where(
        x, ymin=ymin, ymax=ymax, where=np.logical_not(yvalid), facecolor='red', alpha=0.5)
    axis.add_collection(collection)
    return axis


def plot_iv(data, sampling_frequency, show=None):
    x = np.arange(len(data), dtype=float)
    x *= 1.0 / sampling_frequency
    f = plt.figure()

    ax_i = f.add_subplot(2, 1, 1)
    plot_axis(ax_i, x, data[:, 0], label='Current (A)')
    ax_v = f.add_subplot(2, 1, 2, sharex=ax_i)
    plot_axis(ax_v, x, data[:, 1], label='Voltage (V)')

    if show is None or bool(show):
        plt.show()
        plt.close(f)

def str2val (s):
    # Create regex to parse range
    parse_range = re.compile("([0-9]+\.*[0-9]*)([a-zA-Z]+)")    
    res = parse_range.match (s).groups ()
    val = float (res[0])
    unit = res[1].replace ('uA', 'µA')
    if unit == 'mA':
        val /= 1000
    elif unit == 'µA':
        val /= 1000000
    elif unit == 'nA':
        val /= 1000000000
    return [val, str(res[0]) + ' ' + unit]

def switch (js, idx, enable):
    if enable:
        val = '0'
    else:
        val = '1'
    js.parameter_set ('gpo' + str (idx), val)

def dump (args, stat):
    fmt = ticker.EngFormatter()

    if stat['pass']:
        print (Fore.GREEN, end='')
    else:
        print (Fore.RED, end='')
        
    print ('[' + str(stat['dut']) + '] Avg voltage=' + fmt.format_data(stat['v_avg']) + 'V')
    print ('[' + str(stat['dut']) + '] Voltage sag=' + fmt.format_data(stat['v_pp']) + 'V')
    print ('[' + str(stat['dut']) + '] Avg current=' + fmt.format_data(stat['i_avg']) + 'A')
    print ('[' + str(stat['dut']) + '] Max current=' + fmt.format_data(stat['i_max']) + 'A')
    print (Fore.RESET)
        
def record (n, dut, stats, s):
    stats[n]['v_avg'] = s['signals']['voltage']['µ']['value']
    stats[n]['v_pp']  = s['signals']['voltage']['p2p']['value']
    stats[n]['i_avg'] = s['signals']['current']['µ']['value']
    stats[n]['i_max'] = s['signals']['current']['max']['value']
    stats[n]['dut']   = dut
    
def run(args):

    def on_stop(event, message):
        nonlocal quit_
        quit_ = 'quit from stop'

    # Reset DUT
    with scan_require_one(config='auto') as js:
        js.parameter_set('sensor_power', 'on')
        js.parameter_set ('v_range', args.vrange) 
        js.parameter_set('io_voltage', '5.0V')
        if args.delay:
            js.parameter_set ('i_range', 'auto')
        else:
            js.parameter_set ('i_range', args.irange)

        # Turn both off
        switch (js, 0, False)
        switch (js, 1, False)
        time.sleep (0.4)
            
        # Set active DUT
        if args.dut == 'both':
            active = 1

            # Stats to save
            stats = [dict() for x in range (args.cnt * 2)]
        else:
            active = int (args.dut)

            # Stats to save
            stats = [dict() for x in range (args.cnt)]

        # Double cnt if running single unit then
        # every other cycle will be off cycle
        args.cnt *= 2

        # Calculate overload current
        current_overload = args.ilimit if args.delay or (args.irange == 'auto') else args.irange_val
        
        # Loop over range
        for n in range (args.cnt):

            # Turn off inactive DUT
            if args.dut != 'both':
                idx = int(n / 2)
                if (n & 1) == 0:
                    switch (js, active, False)
                    time.sleep (args.off)
                    continue
                else:
                    switch (js, active, True)
            # For both case just toggle
            else:
                idx = n
                
                # Switch off previously active
                switch (js, active, False)
                time.sleep (args.off)
                active = (active + 1) % 2
                    
                # Turn on active DUT
                switch (js, active, True)
                    
            # Delay and do quick read
            data = js.read(contiguous_duration=0.001)
            current, voltage = data[-1, :]

            # Skip if current if too high
            if current >= current_overload:

                # Use eng formatter
                fmt = ticker.EngFormatter()
                
                # Record failure
                stats[idx]['v_avg'] = voltage
                stats[idx]['i_avg'] = current
                stats[idx]['pass'] = False
                stats[idx]['dut'] = active                
                print (Fore.RED + '[' + str(stats[idx]['dut']) + '] Avg current=' +
                       fmt.format_data (current) + 'A LIMIT(' +
                       fmt.format_data (current_overload) + 'A)\n' + Fore.RESET)
                continue
                
            # if delay then sleep and change range
            if args.delay:
                time.sleep (args.delay)
                js.parameter_set ('i_range', args.irange)

            # Start device read
            quit_ = False
            js.start (stop_fn=on_stop, contiguous_duration=args.on)
            #js.start (stop_fn=on_stop, duration=args.on)
                
            # Wait until finished
            while not quit_:
                time.sleep (0.01)

            # Set range back to auto
            if args.delay:
                js.parameter_set ('i_range', 'auto')

            # Get sample ID range
            start, stop = js.stream_buffer.sample_id_range

            # Get samples
            raw = js.stream_buffer.samples_get (start, stop, fields=['current', 'voltage'])
            buf = np.empty((stop-start, 2), dtype=float)
            buf[:] = 0.0
            buf[start:stop, 0] = raw['signals']['current']['value']
            buf[start:stop, 1] = raw['signals']['voltage']['value']

            # Get stats
            s = js.stream_buffer.statistics_get (start, stop)[0]

            # Convert to API format
            t_start = start / js.stream_buffer.output_sampling_frequency
            t_end = stop / js.stream_buffer.output_sampling_frequency
            s = stats_to_api(s, t_start, t_end)

            # Record stats
            record (idx, active, stats, s)

            # Check PASS/FAIL
            if (stats[idx]['i_avg'] < args.min) or (stats[idx]['i_avg'] > args.max):

                # Save failure
                stats[idx]['pass'] = False

                # If break on fail then print
                if args.stop:

                    # Dump
                    dump (args, stats[idx])                    

                    # Plot
                    plot_iv (buf, raw['time']['sampling_frequency']['value'], show=True)
                    break
                    
            else:
                stats[idx]['pass'] = True

            # Dump stats if requested
            if args.stat:
                dump (args, stats[idx])

            # Plot
            if args.plot:
                plot_iv (buf, raw['time']['sampling_frequency']['value'], show=True)

        # Turn both off
        switch (js, 0, False)
        switch (js, 1, False)

        # Display results
        passed = [0, 0]
        failed = [0, 0]
        for stat in stats:
            if 'pass' in stat:
                if stat['pass']:
                    passed[stat['dut']] += 1
                else:
                    failed[stat['dut']] += 1
        if args.dut == 'both':
            for n in range (len(passed)):
                print ('DUT[' + str(n) + '] PASS=' + str(passed[n]) + ' FAIL=' +
                       str(failed[n]) + ' TOTAL=' + str(int(args.cnt / 2)))
        else:
            print ('DUT[' + str(active) + '] PASS=' + str(passed[active]) + ' FAIL=' +
                   str(failed[active]) + ' TOTAL=' + str(int(args.cnt / 2)))
        return 0


if __name__ == '__main__':

    # Create argparse
    parser = argparse.ArgumentParser ()

    # Add arguments
    parser.add_argument ('--on', help='On time in seconds (float)', required=True)
    parser.add_argument ('--off', help='Off time in seconds (float)')
    parser.add_argument ('--delay', help='Delay time in seconds before sampling (float)')
    parser.add_argument ('--min', help='Min acceptable current over window (float)')
    parser.add_argument ('--max', help='Max acceptable current over window (float)')
    parser.add_argument ('--cnt', help='Cycles to run')
    parser.add_argument ('--dut', help='DUT to test 0/1/both (default=both)')
    parser.add_argument ('--stop', help='Stop on first failure', action='store_true')
    parser.add_argument ('--stat', help='Dump stats', action='store_true')
    parser.add_argument ('--plot', help='Plot stats', action='store_true')

    # Parameters for joulescope
    parser.add_argument ('--vrange', help='5V/15V')
    parser.add_argument ('--irange', help='auto/10A/2A/180mA/18mA/1.8mA/180uA/18uA')
    parser.add_argument ('--ilimit', help='Max current to consider overcurrent')
    # Internal only - do not use
    parser.add_argument ('--irange_val', default=0.0, help=argparse.SUPPRESS)
    parser.add_argument ('--irange_unit', default='', help=argparse.SUPPRESS)

    # Parse args
    args = parser.parse_args ()

    # Validate args
    if not args.cnt:
        args.cnt = 1
    if not args.dut:
        args.dut = 'both'
    if not args.vrange:
        args.vrange = '5V'
    if not args.min:
        args.min = '0A'
    if not args.off:
        args.off = 0.2
    if args.max:
        args.max = str2val (args.max)[0]
    if args.ilimit:
        args.ilimit = str2val (args.ilimit)[0]
    else:
        args.ilimit = 0.1 # 100mA default

    args.off = float (args.off)
    args.cnt = int (args.cnt)
    args.min = str2val (args.min)[0]
    args.on = float (args.on)
    if args.delay:
        args.delay = float (args.delay)
    
    # Parse current range - check against min/max
    if not args.irange:
        args.irange = 'auto'
        if not args.max:
            args.max = 10.0
    elif args.irange != 'auto':
        (args.irange_val, args.irange) = str2val (args.irange)
        if not args.max:
            args.max = args.irange_val
        if not args.delay and ((args.max > args.irange_val) or (args.min > args.irange_val)):
            print ('irange invalid with MIN/MAX')
            exit (-1)
            
    # Run test
    run(args)
    
