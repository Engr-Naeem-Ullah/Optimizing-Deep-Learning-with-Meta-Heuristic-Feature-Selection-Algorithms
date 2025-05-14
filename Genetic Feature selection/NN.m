function Yp = NN(trn, l_train, test)
    net = feedforwardnet(10, 'trainlm');
    net = train(net, trn', l_train');
    Y2 = net(test');
    Yp = round(Y2);
end
