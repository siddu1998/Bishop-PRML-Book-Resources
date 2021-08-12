import torch

#create a design matrix with 10 rows and 5 features
#the matrix dimention now will be 10x5
X = torch.arange(50,dtype=torch.float).view(10,5)


#create fully connected linear layer
fc = torch.Linear(in_features=5,
                  out_features=3)

#the weights shape will be? 
# out_features x in_features = 3x5
# bias will always be equal to output size i.e 3

A = fc_layer(X)

#so X : 10x5
#so W : 3x5 
#bias is 3
# hence A will be 10x3
#this means W will be transposed before multiplication

