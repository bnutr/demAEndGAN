import pickle
from output import Output, OutputType, Normalization

data_feature_output = [
	Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False)]

with open('data_feature_output.pkl', 'wb') as f:
    pickle.dump(data_feature_output, f)

