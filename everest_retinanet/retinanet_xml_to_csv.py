'''
    Collate PascalVOC labels into a single CSV. Format requirement:
        - One bounding box per row
        - Each row: file_path,x1,y1,x2,y2,class_name
        - Example:
            /data/imgs/img_001.jpg,837,346,981,456,cow
            /data/imgs/img_002.jpg,22,5,89,84,bird
            /data/imgs/img_003.jpg,,,,,
        - x1,y1,x2,y2 are raw pixel values
        - Note the negative example at the end

    NOTE: must use backslash as path separator, otherwise Tensorflow would go nuts
'''

import os, glob, argparse
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def find_positive_samples(xml_path, output_fname_suffix=''):
    '''
        Read XML files from directory xml_path, and output a single CSV file containing
        file names, bounding boxes and class labels. Use output_fname_suffix to
        indicate relative/absolute paths if desired.
    '''

    xml_list = []
    for xml_file in glob.glob(xml_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (output_fname_suffix.strip('/') + '/' + root.find('filename').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text
                   )
            xml_list.append(value)

    return xml_list


def find_negative_examples(img_dir, xml_dir):
    def _extract_file_name(paths, with_ext=True):
        fnames = [os.path.split(p)[-1] for p in paths]
        if not with_ext:
            fnames = [f.split('.')[0] for f in fnames]
        return fnames

    img_fname = _extract_file_name( glob.glob(f'{img_dir}/*.jpg') , with_ext=False )
    label_fname = _extract_file_name( glob.glob(f'{xml_dir}/*.xml'), with_ext=False )
    negative_samples_fname = set(img_fname).difference(set(label_fname))

    return [ ( img_dir.strip('/')+'/'+fname+'.jpg','','','','','' ) for fname in negative_samples_fname]


def split_df_by_col(df, col, train_frac=0.8):
    '''
        Split a DataFrame into two parts by values in column 'col'.
        train_frac portion of the unique values among 'col' will go to one part, the rest to the other.
    '''

    assert 0. < train_frac < 1.0, "train_frac must be between 0 and 1"

    all_vals = df[col].unique()
    train_vals = np.random.choice( all_vals, replace=False, size=int( len(all_vals)*train_frac ) )
    test_vals = set(all_vals).difference(set(train_vals))

    df_copy = df.set_index(col)
    train_df = df_copy.loc[train_vals].reset_index()
    test_df = df_copy.loc[test_vals].reset_index()

    return train_df, test_df


def save_csv(df, fpath):
    df.to_csv(fpath, index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imgs_dir", help="Directory containing images", type=str)
    parser.add_argument("labels_dir", help="Directory containing labels", type=str)
    # parser.add_argument("--save_to_path", help="Path to output CSV file;", type=str)
    parser.add_argument("--train_test_split", help="Whether to perform train-test split", action="store_true")
    parser.add_argument("--seed", help="Integer for setting random seed in NumPy", type=int)
    args = parser.parse_args()

    ## Define XML and images directories
    imgs_dir, labels_dir = args.imgs_dir, args.labels_dir

    if args.seed:
        np.random.seed(args.seed)
    else:
        np.random.seed(1234252) # For DEBUG purposes, fixing the seed

    # Form lists of 5-tuples: (filename, xmin, ymin, xmax, ymax, class)
    positive_samples = find_positive_samples(labels_dir, output_fname_suffix=imgs_dir)
    negative_samples = find_negative_examples(imgs_dir, labels_dir)

    # Put together a DataFrame containing all data
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class_label']
    full_df = pd.DataFrame(positive_samples + negative_samples, columns=column_name)

    # Split into train and test sets if instructed
    if args.train_test_split:
        train_df, test_df = split_df_by_col(full_df, col='filename', train_frac=0.8)
        assert (full_df['class_label'].value_counts() == \
                (train_df['class_label'].value_counts() + test_df['class_label'].value_counts())).all(), "Lost Something?"
        save_csv(train_df, f"train.csv")
        save_csv(test_df, f"test.csv")
        print("Saved training and test DataFrames")
    else:
        save_csv(full_df, "full.csv")
        print("Saved full DataFrame; no train-test splitting")

    # Define XML and images directories
    # Potential argparse objects
    # labels_dir = 'labels'
    # imgs_dir = 'imgs'
    # np.random.seed(1234252)
    #
    # # Form lists of 5-tuples: (filename, xmin, ymin, xmax, ymax, class)
    # positive_samples = find_positive_samples(labels_dir, output_fname_suffix=imgs_dir)
    # negative_samples = find_negative_examples(imgs_dir, labels_dir)
    #
    # # Put together a DataFrame containing all data
    # column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class_label']
    # full_df = pd.DataFrame(positive_samples + negative_samples, columns=column_name)
    #
    # # Split into train and test sets
    # train_df, test_df = split_df_by_col(full_df, col='filename', train_frac=0.8)
    # assert (full_df['class_label'].value_counts() == \
    #         (train_df['class_label'].value_counts() + test_df['class_label'].value_counts())).all(), "Lost Something?"
    #
    # # Save train and test sets to file
    # save_csv(train_df, 'train.csv')
    # save_csv(test_df, 'test.csv')
    #
    # print('Successfully converted xml to csv.')

if __name__ == '__main__':
    main()
