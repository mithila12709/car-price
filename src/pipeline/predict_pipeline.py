import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # for loading pkl file


class PredictPipeline:
    def __init__(self):
        pass

    # This func nothing but model prediction file
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
        # custom data class is responsible for mapping all the inputs we are giving in the HTML
class CustomData:
     def __init__(self,
        Car_Name: str,
        Year: int,
        Present_Price: int,
        Kms_Driven: int,
        Seller_Type: int,
        Transmission: int,
        Owner: int):
       
        
        self.Car_Name=Car_Name
        self.Year = Year
        self.Present_Price= Present_Price
        self.Kms_Driven=Kms_Driven
        self.Seller_Type=Seller_Type
        self.Transmission= Transmission
        self.Owner=Owner
        

# this function will return all my input in the form of data frame.
    
     def get_data_as_data_frame(self):
          try:
            custom_data_input_dict = {
               " Car_Name":[self.Car_Name],
                "Year":[self.Year],
                "Present_Price":[self.Present_Price],
                "Kms_Driven":[self.Kms_Driven],
                "Seller_Type":[self.Seller_Type],
                "Transmission":[self.Transmission],
                "Owner":[self.Owner]    
            }

            return pd.DataFrame(custom_data_input_dict)

          except Exception as e:
            raise CustomException(e, sys)
                
