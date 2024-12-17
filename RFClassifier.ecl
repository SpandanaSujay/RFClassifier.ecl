IMPORT ML_Core;
IMPORT LearningTrees;

MyRecord := RECORD
    STRING6 gender;           
    INTEGER age;
    INTEGER hypertension;
    INTEGER heart_disease;
    STRING12 smoke;  
    DECIMAL4_2 bmi;
    DECIMAL2_1 HbA1c_level;
    INTEGER blood_glucose_level;
    UNSIGNED1 diabetes; 
END;

MyDataset := DATASET('~spandana::internship::diabetes_prediction_dataset.csv', MyRecord, CSV(HEADING(1)));


IDRec := RECORD
    UNSIGNED   RecID;          
    UNSIGNED1  gender;           
    INTEGER    age;
    INTEGER    hypertension;
    INTEGER    heart_disease;
    UNSIGNED1  smoke;  
    DECIMAL4_2 bmi;
    DECIMAL2_1 HbA1c_level;
    INTEGER    blood_glucose_level;
    UNSIGNED1  diabetes; 
END;    

OUTPUT(MyDataset, NAMED('InputDataset'));

recordCount := COUNT(MyDataset);
OUTPUT(recordCount, NAMED('recordCount'));

splitRatio := 0.8;

CleanRec := RECORD
    IDRec;
    UNSIGNED4 rnd;
END;

newDs := PROJECT(MyDataset, TRANSFORM(CleanRec,
                                      SELF.recid  := COUNTER,
                                      SELF.rnd    := RANDOM(),
                                      SELF.gender := CASE(LEFT.Gender,'Male' => 1,'Female' => 2,'Other' => 3,0),
                                      SELF.smoke  := CASE(LEFT.Smoke,'No Info' => 1,'current' => 2,'not' => 3,'former' => 4,'never' => 5,'ever' => 6,0),
                                      SELF        := LEFT));
shuffledDs := SORT(newDs, rnd);
CleanDS := PROJECT(shuffledDs,IDRec);

TrainDs := CleanDs[1..(recordCount * splitRatio)];
TestDs  := CleanDs[(recordCount*splitRatio + 1)..recordCount];

OUTPUT(TrainDs, NAMED('TrainDataset'));
OUTPUT(TestDs, NAMED('TestDataset'));

ML_Core.ToField(TrainDS, TrainNF);
ML_Core.ToField(TestDS, TestNF);

OUTPUT(TrainNF, NAMED('TrainNumericField'));
OUTPUT(TestNF, NAMED('TestNumericField'));

independent_cols := 8;

X_train := TrainNF(number < independent_cols + 1);
y_train := ML_Core.Discretize.ByRounding(TrainNF(number = independent_cols + 1));
NTrain := PROJECT(y_Train,TRANSFORM(RECORDOF(y_train),SELF.number := 1,SELF := LEFT));

X_test := TestNF(number < independent_cols + 1);
y_test := ML_Core.Discretize.ByRounding(TestNF(number = independent_cols + 1));
NTest := PROJECT(y_Test,TRANSFORM(RECORDOF(y_test),SELF.number := 1,SELF := LEFT));

OUTPUT(X_train,NAMED('x_train'));
OUTPUT(Ntrain,NAMED('y_train'));
OUTPUT(X_test,NAMED('x_test'));
OUTPUT(NTest,NAMED('y_test'));

F := LearningTrees.ClassificationForest(); 

classifier := F.GetModel(X_train, Ntrain);

predicted  := F.Classify(classifier, X_test);

OUTPUT(predicted, NAMED('PredictedY'));
OUTPUT(NTest, NAMED('ActualY'));

cm := ML_Core.Analysis.Classification.ConfusionMatrix(predicted, NTest);
OUTPUT(cm, NAMED('ConfusionMatrix'));

accuracy_values := ML_Core.Analysis.Classification.Accuracy(predicted, NTest);

OUTPUT(accuracy_values, NAMED('AccuracyValues'));