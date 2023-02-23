import gradio as gr
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os

def process_input(input_df):
    # deal with na values
    X = input_df.drop(['annotation_atomrec'], axis=1)
    
    # encode the categorical values
    encoded_X = X
    encoder = LabelEncoder()
    encoder.fit(X['annotation_sequence'])
    encoded_X['annotation_sequence'] = encoder.transform(X['annotation_sequence'])
    
    encoder.fit(X['entry'])
    encoded_X['entry'] = encoder.transform(X['entry'])
    
    # scale the numeric
    
    scaled_X = MinMaxScaler().fit_transform(encoded_X)
    scaled_X[: , 0] = encoded_X['annotation_sequence']
    scaled_X[: , -2] = encoded_X['entry']
    scaled_X[: , -1] = encoded_X['entry_index']
    
    return scaled_X

input_labels = ['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F',
       'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N',
       'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W',
       'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU',
       'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H',
       'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I',
       'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7',
       'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11',
       'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z',
       'entry', 'entry_index']

inputs = [gr.Dataframe(row_count = (1, "dynamic"), col_count=(49,"dynamic"), label=input_labels, interactive=1)]
outputs = [gr.Dataframe(row_count = (1, "dynamic"), col_count=(1, "fixed"), label="Predictions", headers=["False/True"])]
model = pickle.load(open(os.path.join("Cyclica", "kebing_model.sav"), 'rb'))

df = pd.read_csv(os.path.join("Cyclica", "af2_dataset_training_labeled.csv.gz"), index_col=0)
df = df.drop(['y_Ligand'], axis=1)

def infer(input_dataframe):
  X = process_input(input_dataframe)
  predictions = model.predict(X)
  return pd.DataFrame(predictions)

gr.Interface(fn = infer, inputs = inputs, outputs=outputs,
             title="residue (row) belongs to a known binding site? or not?",
             examples = [df.head(2)]).launch()
