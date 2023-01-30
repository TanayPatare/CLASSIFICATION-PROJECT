def model_builder():
#Importing Libraries
        import numpy as np              
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib

#Loading Dataset
        df = pd.read_csv("heart.csv")

#Differentiating categorical and numerical variables
        categorical_val = []
        continous_val = []
        for column in df.columns:
            if len(df[column].unique()) <= 10:
                categorical_val.append(column)
            else:
                continous_val.append(column)
    
        categorical_val.remove("target")
        dataset = pd.get_dummies(df, columns = categorical_val)

#Splitting the Data
        X_train,X_test,Y_train,Y_test = train_test_split(dataset,df["target"],test_size=0.3)

#Model building
        model_rf = RandomForestClassifier()
        model_knn = KNeighborsClassifier(n_neighbors =7, metric="minkowski",p=2)
        model_lr = LogisticRegression()
        model_svc = SVC()

#Training model
        model_rf.fit(X_train, Y_train)
        model_knn.fit(X_train, Y_train)
        model_lr.fit(X_train, Y_train)
        model_svc.fit(X_train, Y_train)

#Prdeictions
        y_pred_rf = model_rf.predict(X_test)
        y_pred_knn = model_knn.predict(X_test)
        y_pred_lr = model_lr.predict(X_test)
        y_pred_svc = model_svc.predict(X_test)

#Testing the accuracy of model
        acc_thres = 95
        acc_rf = accuracy_score(Y_test, y_pred_rf)
        acc_knn = accuracy_score(Y_test, y_pred_knn)
        acc_lr = accuracy_score(Y_test, y_pred_lr)
        acc_svc = accuracy_score(Y_test, y_pred_svc)

        #joblib.dump(model_rf,r"E:/PROJECTS/HEART-ATTACK-PREDICTION-1/model.pkl")

#List of Accuracy
        acc_list = [acc_rf,acc_knn,acc_lr,acc_svc]

#Getting best Model
        if acc_rf > acc_thres:
            best_model_index = acc_list.index((max(acc_list)))
            mod_list = [model_rf,model_knn,model_lr,model_svc]
            #Dumping Model
            joblib.dump(mod_list[best_model_index],r"E:/PROJECTS/HEART-ATTACK-PREDICTION-1/model.pkl")
            return model_rf
        
        else:
            return "Model is low, please retrain the model"
        
        