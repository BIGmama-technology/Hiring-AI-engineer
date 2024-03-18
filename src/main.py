from models.train_bnn import *
# ------------------------------------------
# mauna_loa_atmospheric_co2 Dataset
# ------------------------------------------
run(model1,neg_log_likelihood,train_dataloader_1,test_dataloader_1,X_test,num_epochs,learning_rate)
# ------------------------------------------
# international-airline-passengers Dataset
# ------------------------------------------
run(model2,neg_log_likelihood,train_dataloader_2,test_dataloader_2,X_test,num_epochs,learning_rate)

# export models
torch.save(model1,"./models/mauna_loa_model.pth")
torch.save(model2, "./models/international_airline_passengers_model.pth")
