function Yp = NB(train, l_train, test)
    model = fitcnb(train, l_train);
    Yp = predict(model, test);
end
