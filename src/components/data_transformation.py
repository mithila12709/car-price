import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
   def __init__(self):
      self.data_transformation_config=DataTransformationConfig()

   def get_data_transformer_obj(self):
      # this function responsible for data transformation
      try:
         numarical_col=["Year","Present_Price","Kms_Driven"]
         cat_col=["Fuel_Type","Seller_Type","Transmission","Car_Name"]

         num_pipeline=Pipeline(
            steps=[
               ("imputer",SimpleImputer(strategy="median"))
               #("scaler", StandardScaler())
            ]
         )
         cat_pipeline=Pipeline(
            steps=[
               ("imputer",SimpleImputer(strategy="most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown='ignore'))
               #("scaler", StandardScaler())
            ]
         )
         logging.info("numarical columns scaling completed")

         logging.info("categorical columns encoding completed")

         preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numarical_col),
            ("cat_pipeline",cat_pipeline,cat_col)
            ]
         )
         return preprocessor
         
      except Exception as e:
         raise CustomException(e,sys)
      
   def initiate_data_transformation(self,train_path,test_path):
      try:
         train_df=pd.read_csv(train_path)
         test_df=pd.read_csv(test_path)

         logging.info("Read train and test data completed")

         preprocessor_obj=self.get_data_transformer_obj()

         target_column_name="Selling_Price"
         numarical_col=["Year","Present_Price","Kms_Driven"]

         input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
         target_feature_train_df=train_df[target_column_name].values.reshape(-1, 1)

         input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
         target_feature_test_df=test_df[target_column_name].values.reshape(-1, 1)

          #  shapes of dataframes before transformation
         print("input_feature_train_df",input_feature_train_df.shape)
         print("target_feature_train_df",target_feature_train_df.shape)
         print("input_feature_test_df",input_feature_test_df.shape)
         print("target_feature_test_df",target_feature_test_df.shape)

         input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
         input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

         #  shapes of arrays after transformation
         print("input_feature_train_arr_after_transformation",input_feature_train_arr.shape)
         print("input_feature_test_arr_after_transformation",input_feature_test_arr.shape)

          # Verify the shapes before concatenation
         print("input_feature_train_arr",input_feature_train_arr.shape)
         print("target_feature_train_df",target_feature_train_df.shape)
         print("input_feature_test_arr",input_feature_test_arr.shape)
         print("target_feature_test_df",target_feature_test_df.shape)

         # Verify the type of the arrays
         print(type(input_feature_train_arr))
         print(type(target_feature_train_df))
         print(type(input_feature_test_arr))
         print(type(target_feature_test_df))

         input_feature_train_arr = input_feature_train_arr.toarray() if not isinstance(input_feature_train_arr, np.ndarray) else input_feature_train_arr
         input_feature_test_arr = input_feature_test_arr.toarray() if not isinstance(input_feature_test_arr, np.ndarray) else input_feature_test_arr

         # Verify if the arrays are truly two-dimensional
         print(input_feature_train_arr.ndim)
         print(target_feature_train_df.ndim)
         print(input_feature_test_arr.ndim)
         print(target_feature_test_df.ndim)




         # train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df) ]

         # test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
         train_arr = np.concatenate([input_feature_train_arr, target_feature_train_df], axis=1)
         test_arr = np.concatenate([input_feature_test_arr, target_feature_test_df], axis=1)

         # Log shapes of final arrays
         print(train_arr.shape)
         print(test_arr.shape)

         logging.info("Saved preprocessing obj")

         save_object(         # used for saving the pkl file
            
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor_obj
         )
         return(
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,

         )
      except Exception as e:
         raise CustomException(e,sys)
         
    
      
   