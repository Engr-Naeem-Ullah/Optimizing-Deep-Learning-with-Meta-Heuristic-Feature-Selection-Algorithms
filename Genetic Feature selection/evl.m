function out=evl(data,labels,algorithm)
    original_data = data;
    original_label = labels;
    num_data = size(data, 1); % Total number of data points
    ten = min(round(num_data/10), num_data); % Ensure ten is not larger than the number of data points
   % fprintf('Number of data points: %d\n', num_data); % Debugging output
    %fprintf('Value of ten: %d\n', ten); % Debugging output
    for i = 1:10   
        data = original_data;
        label = original_label;
        test_indices = ((i-1)*ten+1 : min(i*ten, num_data)); % Ensure test indices do not exceed array bounds
        test = data(test_indices, :);
        l_test = label(test_indices, :);
        data(test_indices, :) = [];
        label(test_indices, :) = [];
        train = data;
        l_train = label;
        switch algorithm
            case 'NB'
                Class = NB(train, l_train, test);
            case 'KNN'
                Class = KNN(train, l_train, test);
            case 'DT'
                Class = DT(train, l_train, test);
            case 'NN'
                Class = NN(train, l_train, test); 
        end
        accuracy(i) = evaluation(Class, l_test);
    end
    out = mean(accuracy) * 100;
end
