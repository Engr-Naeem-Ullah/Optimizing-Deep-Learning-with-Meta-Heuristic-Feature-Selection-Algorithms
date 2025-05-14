function Yp = KNN(train, l_train, test)
    mdl = ClassificationKNN.fit(train, l_train);
    Yp = predict(mdl, test);
end
