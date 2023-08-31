import pickle
import uvicorn
import cloudpickle
import numpy as np
import pandas as pd
import catboost as cb
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
  """Preprocesses the input DataFrame by converting categorical columns to numerical values."""
  df = df[['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
       'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
       'OWN_CAR_AGE', 'OCCUPATION_TYPE', 'REGION_RATING_CLIENT',
       'WEEKDAY_APPR_PROCESS_START', 'REG_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
       'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'COMMONAREA_AVG',
       'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'ENTRANCES_MODE',
       'NONLIVINGAREA_MODE', 'FLOORSMAX_MEDI', 'LIVINGAREA_MEDI',
       'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE',
       'EMERGENCYSTATE_MODE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']]
  
  bureau = pd.read_csv(f'additional_data/bureau.csv')
  credit_card_balance = pd.read_csv(f'additional_data/credit_card_balance.csv')
  installments_payments = pd.read_csv(f'additional_data/installments_payments.csv')
  previous_application = pd.read_csv(f'additional_data/previous_application.csv')
  POS_CASH_balance = pd.read_csv(f'additional_data/POS_CASH_balance.csv')
  
  df = df.merge(bureau, how='left', on='SK_ID_CURR')\
      .merge(credit_card_balance, how='left', on='SK_ID_CURR')\
      .merge(installments_payments, how='left', on='SK_ID_CURR')\
      .merge(previous_application, how='left', on='SK_ID_CURR')\
      .merge(POS_CASH_balance, how='left', on='SK_ID_CURR')
      
  for col in categorical_cols:
      df[col] = df[col].astype('category')
      
  return df

categorical_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

app = FastAPI()
with open('preprocessor.pkl', 'rb') as f:
    preprocessor_catboost = cloudpickle.load(f)
model = load('catboost_model.joblib')

class InputData(BaseModel):
  SK_ID_CURR: int
  NAME_CONTRACT_TYPE: str
  CODE_GENDER: str
  FLAG_OWN_CAR: str
  FLAG_OWN_REALTY: str
  CNT_CHILDREN: int
  AMT_INCOME_TOTAL: float
  AMT_CREDIT: float
  AMT_ANNUITY: float
  AMT_GOODS_PRICE: float
  NAME_TYPE_SUITE: str
  NAME_INCOME_TYPE: str
  NAME_EDUCATION_TYPE: str
  NAME_FAMILY_STATUS: str
  NAME_HOUSING_TYPE: str
  REGION_POPULATION_RELATIVE: float
  DAYS_BIRTH: int
  DAYS_EMPLOYED: int
  DAYS_REGISTRATION: float
  DAYS_ID_PUBLISH: int
  OWN_CAR_AGE: float
  FLAG_MOBIL: int
  FLAG_EMP_PHONE: int
  FLAG_WORK_PHONE: int
  FLAG_CONT_MOBILE: int
  FLAG_PHONE: int
  FLAG_EMAIL: int
  OCCUPATION_TYPE: str
  CNT_FAM_MEMBERS: float
  REGION_RATING_CLIENT: int
  REGION_RATING_CLIENT_W_CITY: int
  WEEKDAY_APPR_PROCESS_START: str
  HOUR_APPR_PROCESS_START: int
  REG_REGION_NOT_LIVE_REGION: int
  REG_REGION_NOT_WORK_REGION: int
  LIVE_REGION_NOT_WORK_REGION: int
  REG_CITY_NOT_LIVE_CITY: int
  REG_CITY_NOT_WORK_CITY: int
  LIVE_CITY_NOT_WORK_CITY: int
  ORGANIZATION_TYPE: str
  EXT_SOURCE_1: float
  EXT_SOURCE_2: float
  EXT_SOURCE_3: float
  APARTMENTS_AVG: float
  BASEMENTAREA_AVG: float
  YEARS_BEGINEXPLUATATION_AVG: float
  YEARS_BUILD_AVG: float
  COMMONAREA_AVG: float
  ELEVATORS_AVG: float
  ENTRANCES_AVG: float
  FLOORSMAX_AVG: float
  FLOORSMIN_AVG: float
  LANDAREA_AVG: float
  LIVINGAPARTMENTS_AVG: float
  LIVINGAREA_AVG: float
  NONLIVINGAPARTMENTS_AVG: float
  NONLIVINGAREA_AVG: float
  APARTMENTS_MODE: float
  BASEMENTAREA_MODE: float
  YEARS_BEGINEXPLUATATION_MODE: float
  YEARS_BUILD_MODE: float
  COMMONAREA_MODE: float
  ELEVATORS_MODE: float
  ENTRANCES_MODE: float
  FLOORSMAX_MODE: float
  FLOORSMIN_MODE: float
  LANDAREA_MODE: float
  LIVINGAPARTMENTS_MODE: float
  LIVINGAREA_MODE: float
  NONLIVINGAPARTMENTS_MODE: float
  NONLIVINGAREA_MODE: float
  APARTMENTS_MEDI: float
  BASEMENTAREA_MEDI: float
  YEARS_BEGINEXPLUATATION_MEDI: float
  YEARS_BUILD_MEDI: float
  COMMONAREA_MEDI: float
  ELEVATORS_MEDI: float
  ENTRANCES_MEDI: float
  FLOORSMAX_MEDI: float
  FLOORSMIN_MEDI: float
  LANDAREA_MEDI: float
  LIVINGAPARTMENTS_MEDI: float
  LIVINGAREA_MEDI: float
  NONLIVINGAPARTMENTS_MEDI: float
  NONLIVINGAREA_MEDI: float
  FONDKAPREMONT_MODE: str
  HOUSETYPE_MODE: str
  TOTALAREA_MODE: float
  WALLSMATERIAL_MODE: str
  EMERGENCYSTATE_MODE: str
  OBS_30_CNT_SOCIAL_CIRCLE: float
  DEF_30_CNT_SOCIAL_CIRCLE: float
  OBS_60_CNT_SOCIAL_CIRCLE: float
  DEF_60_CNT_SOCIAL_CIRCLE: float
  DAYS_LAST_PHONE_CHANGE: float
  FLAG_DOCUMENT_2: int
  FLAG_DOCUMENT_3: int
  FLAG_DOCUMENT_4: int
  FLAG_DOCUMENT_5: int
  FLAG_DOCUMENT_6: int
  FLAG_DOCUMENT_7: int
  FLAG_DOCUMENT_8: int
  FLAG_DOCUMENT_9: int
  FLAG_DOCUMENT_10: int
  FLAG_DOCUMENT_11: int
  FLAG_DOCUMENT_12: int
  FLAG_DOCUMENT_13: int
  FLAG_DOCUMENT_14: int
  FLAG_DOCUMENT_15: int
  FLAG_DOCUMENT_16: int
  FLAG_DOCUMENT_17: int
  FLAG_DOCUMENT_18: int
  FLAG_DOCUMENT_19: int
  FLAG_DOCUMENT_20: int
  FLAG_DOCUMENT_21: int
  AMT_REQ_CREDIT_BUREAU_HOUR: float
  AMT_REQ_CREDIT_BUREAU_DAY: float
  AMT_REQ_CREDIT_BUREAU_WEEK: float
  AMT_REQ_CREDIT_BUREAU_MON: float
  AMT_REQ_CREDIT_BUREAU_QRT: float
  AMT_REQ_CREDIT_BUREAU_YEAR: float

@app.get('/')
def index():
  return {'message': 'It works.'}

@app.post("/predict")
def predict(input_data: InputData):
  X = pd.DataFrame(input_data.dict(by_alias=True), index=[0])
  X = preprocessing(X)
  numerical_cols = [var for var in X.columns if var not in categorical_cols]
  
  X_test_preprocessed = preprocessor_catboost.transform(X)
  X_test_preprocessed = pd.DataFrame(X_test_preprocessed, 
                                     columns = numerical_cols + categorical_cols)
  cat_features_index = [i for i, col in enumerate(X_test_preprocessed.columns) if col in categorical_cols]
  test_pool = cb.Pool(data=X_test_preprocessed, cat_features=cat_features_index) 
  
  y_pred = model.predict(test_pool)[0]
  prediction_mapping = {0: 'No payment difficulties', 1: 'Payment difficulties'}
  prediction = prediction_mapping.get(y_pred, 'Error')
  return {'prediction': prediction}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)

# python -m uvicorn main:app --reload