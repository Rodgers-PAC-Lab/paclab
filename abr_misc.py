"""Functions used for ABR analysis (not in paper)"""

import numpy as np
import pandas
import socket

def get_metadata(data_directory, datestring, metadata_version):
    """Get the metadata from a day of recordings

    TODO: Remove the metadata versioning system and simplify this function

    Looks for the metadata csv file
    Reads this csv file
    Forms a recording_name column
    Forms a datafile column

    Parameters:
        data_directory: where the metadata and the subfolders for recordings live
        datestring: yymmdd datestring
        metadata_version: which version of metadata csv is used (different columns and naming schemes)
    Returns: Metadata as dataframe, with datafile (file path) column added
    """

    # Form the filename to the csv file
    if metadata_version == "v4":
        csv_filename = os.path.join(
            data_directory,
            datestring + '_notes_v4.csv')
    elif metadata_version == "v5":
        csv_filename = os.path.join(
            data_directory,
            datestring + '_notes_v5.csv')
    elif metadata_version == "v6":
        csv_filename = os.path.join(
            data_directory,
            datestring + '_notes_v6.csv')
    else:
        print("ERROR, Metadata csv not found for " + datestring)
        return

    # Read the CSV file
    metadata = pandas.read_csv(csv_filename)
    metadata['include'] = metadata['include'].fillna(1).astype(bool)

    # Deserialize the amplitudes with json
    metadata['amplitude'] = metadata['amplitude'].apply(json.loads)

    # Form ch0_config and make recording_name from session number
    if metadata_version=='v4':
        metadata['recording_name'] = ['{:03d}'.format(n) for n in metadata['session'].values]
        metadata['ch0_config'] = (metadata['positive_electrode'].str.cat(metadata['negative_electrode']))
        metadata['ch2_config'] = (metadata['positive_ch3'].str.cat(metadata['negative_ch3']))
        metadata['ch4_config'] = 'NN'
        metadata = metadata.rename(columns={"session":"recording"})
        metadata = metadata.drop(['pre_vs_post','positive_electrode','negative_electrode','positive_ch3','negative_ch3'],axis=1)
    elif metadata_version=='v5' or metadata_version=='v6':
        metadata['recording_name'] = ['{:03d}'.format(n) for n in metadata['recording'].values]
        if metadata_version=='v5':
            metadata['ch4_config'] = 'NN'
    else:
        print("ERROR, METADATA VERSION NOT V4, V5, OR V6")
        return
    
    # Form the full path to the session
    metadata['datafile'] = [
        os.path.join(data_directory, recording_name)
        for recording_name in metadata['recording_name']]
    
    return metadata

def drop_torn_packets(data,header_df, debug_plot=False,speaker_channel=7):
    # Go through and find where the tearing is
    torn = np.diff(header_df['packet_num_unwrapped']) != 1
    # This labels data as torn ON THE PACKET WHERE IT MESSED UP, instead of the next correct packet
    header_df['torn'] = np.array([*torn, False])
    # To calculate the correct sample numbers we need to know the number of packets recieved by the GUI,
    #  rather than the ACTUAL packet number sent from the Teensy
    header_df['pkt_sequence'] = np.arange(len(header_df))
    header_df['sample_number_start'] = header_df['pkt_sequence'] * 500
    torn_headers = header_df.loc[header_df['torn'] == True]

    # Optional plot of the speaker channel with torn packets highlighted in green
    if debug_plot:
        import matplotlib.pyplot as plt
        plt.plot(data[:, speaker_channel])
        bad_samples = []
        for pkt in torn_headers.index:
            start = torn_headers.loc[pkt]['sample_number_start']
            bad_samples.append([start, start + 500])
            plt.axvspan(start, start + 500, facecolor='green', alpha=0.5)
        plt.show()

    packets_l = []
    for pkt_n in range(len(header_df)):
        pkt = header_df.iloc[pkt_n]
        if pkt['torn'] == False:
            packets_l.append(
                data[(pkt_n * 500): (pkt_n + 1) * 500]
            )
    new_data = np.concatenate(packets_l)

    # Make sure the number of samples dropped matches the number of packets torn
    assert len(data) - len(new_data) == len(torn_headers) * 500

    return new_data, header_df

def get_ABR_data_paths():
    """Set directories for ABR analysis separately for each computer
    
    This script gets the hostname and returns directories to use for
    ABR data and data storage.
    
    Returns: GUIdata_directory, Pickle_directory
        GUIdata_directory : path to location of raw ABR data
        Pickle_directory : path to location where intermediate data files
            and figures can be stored
    """
    # Get the hostname
    computer = socket.gethostname()
    
    # Appropriately set the directories based on hostname
    if computer == 'squid':
        # Rowan's lab desktop
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.expanduser('~/mnt/cuttlefish/rowan/ABR/Figs_Pickles')
    
    elif computer == 'mantaray':
        # The surgery computer- windows partition
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            '~/Pickle_Temporary_Storage'))
    
    elif computer == 'NSY-0183-PC':
        # Rowan's new laptop
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/kgargiu/Documents/GitHub/pickles'))
        Pickle_directory = GUIdata_directory
    
    elif computer=='Theseus':
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            'C:/Users/kgarg/Documents/GitHub/pickles'))
        Pickle_directory =  GUIdata_directory

    elif computer=='kelp':
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.expanduser(
        '~/mnt/cuttlefish/cedric/data/ABR/pickles')
    
    elif computer == 'cephalopod':
        # Chris' computer
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/chris/data/20250720_abr_data'))
            
    else:
        # defaults
        GUIdata_directory = os.path.normpath(os.path.expanduser(
            '~/mnt/cuttlefish/surgery/abr_data'))
        Pickle_directory = os.path.normpath(os.path.expanduser(
            '~/Pickle_Temporary_Storage'))
    
    return GUIdata_directory, Pickle_directory

def laterality_check(channel, speaker_side):
    """
    Determine if the sound is playing on the contralateral or ipsilateral side from the recording on a particular channel.
    Only works with channels LV and RV.
    LR doesn't really have an ipsilateral or contralateral side since it records from both ears.

    Parameters:
        channel: string 'LV' or 'RV', which channel the data is from
        speaker_side: string 'L' or 'R', which side the speaker plays on

    Returns:
        laterality: string 'ipsilateral' or 'contralateral'. Returns np.nan if it gets invalid input.
    """
    if speaker_side == 'L':
        if channel == 'LV':
            laterality = 'ipsilateral'
        elif channel == 'RV':
            laterality = 'contralateral'
        else:
            print("Invalid channel and speaker side config: ",channel,speaker_side)
            laterality = np.nan
    elif speaker_side == 'R':
        if channel == 'RV':
            laterality = 'ipsilateral'
        elif channel == 'LV':
            laterality = 'contralateral'
        else:
            print("Invalid channel and speaker side config: ", channel, speaker_side)
            laterality = np.nan
    else:
        print("Invalid channel and speaker side config: ", channel, speaker_side)
        laterality = np.nan
    return laterality

def pre_or_post(timepoint):
    """
    Determine if the timepoint is pre or post hearing loss.
    Timepoints are usually labeled like apreA, postB, etc.

    Parameters:
        timepoint: string of timepoint to check

    Returns:
        res: string 'pre' or 'post'. Returns np.nan if timepoint doesn't contain 'pre' or 'post'
    """
    if 'pre' in timepoint:
        res = 'pre'
    elif 'post' in timepoint:
        res = 'post'
    else:
        print('ERROR, ' + str(timepoint) + 'not pre or post!')
        res = np.nan
    return res

def join_cohort_info_to_df(df, cohort_experiments, join_on=['date','mouse'],
                           join_HL=True, join_timepoint=True, join_sex=False,
                           join_strain=False, join_genotype=False, join_age=False):
    """
    Takes any df that includes mouse and date, then adds useful info from cohort_experiments

    Arguments:
        df: pandas Dataframe with mouse and date somewhere in the index or on the columns.
        cohort_experiments: pandas Dataframe that's based on mouse_info.csv
        join_on: list of str, the columns that are the same in both dataframes which you join on.
            Pretty much always should be 'date' and 'mouse'
        join_HL: boolean, do you want to add hearing loss status to df
        join_timepoint: boolean, do you want to add testing timepoint (pre-HL, post-HL, etc) to df
        join_sex: boolean, do you want to add mouse sex to df
        join_strain: boolean, do you want to add mouse strain to df
        join_genotype: boolean, do you want to add mouse genotype to df
        join_age: boolean, do you want to add mouse age to df
    Returns:
        df: pandas Dataframe
            The original dataframe with your chosen columns added. Keeps its original index.
    """
    if 'timepoint' in join_on and join_timepoint==True:
        print("You're trying to join on timepoint but also append timepoint, which won't work.")
        print("Changing join_timepoint==False so the join works. Timepoint will be in index not columns.")
        join_timepoint=False
    columns_ser = pandas.Series({
        'HL':   join_HL,
        'timepoint': join_timepoint,
        'sex': join_sex,
        'strain': join_strain,
        'genotype': join_genotype,
        'age': join_age
    })

    # Use booleans to get a list of the join_ columns you set True
    columns_to_join = columns_ser.loc[columns_ser==True].index
    # Check to see if you're trying to join a column that already exists in df
    df_idx_l = df.index.names
    for col in columns_to_join:
        if col in df_idx_l:
            columns_to_join = columns_to_join.drop(col)
            print('Tried to join  the column ' + col + ', but it already exists in the original df')


    # Save the old index to re-apply later
    df_idx = df.index.to_frame()
    # Change index of df and cohort_experiments to match
    df = df.reset_index().set_index(join_on)
    cohort_experiments = cohort_experiments.reset_index().set_index(join_on)

    # Join selected columns from cohort_experiments to df
    df = df.join(cohort_experiments[columns_to_join], on=join_on)
    # Set the df index back how it was
    df.index = pandas.MultiIndex.from_frame(df_idx)
    return df

def plot_single_ax_abr(abr_subdf, ax, sampling_rate=16000):
    """
    PARAMETERS:
        abr_subdf: a subdf where the index is sound levels and the columns are voltages
        ax: the axis to plot it on
        t: the x axis in ms
    RETURNS:
        ax: the axis object with the plot made
    """
    t = abr_subdf.columns/sampling_rate*1000
    for label_i in abr_subdf.index.sort_values(ascending=False):
        aut_colorbar = generate_colorbar(
            len(abr_subdf.index), mapname='inferno_r', start=0.15, stop=1)
        color_df = pandas.DataFrame(aut_colorbar,
                                    index=abr_subdf.index.sort_values(ascending=True))
        ax.plot(t, abr_subdf.loc[label_i].T * 1e6, lw=.75,
                color=color_df.loc[label_i], label=label_i)
    return ax