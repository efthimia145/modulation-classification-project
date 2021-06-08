function merge_mat_files(NTrain, L, NClasses, files_path, postfix_data, postfix_label, ...
                            train_flag, SNR_low, SNR_high, SNR_step)

    NSNRs = ((SNR_high - SNR_low)/SNR_step) + 1;

    if train_flag == 1
        train_data = zeros(NTrain*NClasses*NSNRs, L);
        train_label = zeros(NTrain*NClasses*NSNRs, 1);
    else
        test_data = zeros(NTrain*NClasses*NSNRs, L+2);
    end

    counter_N = -1;

    for count_classes = 0:NClasses-1
        for SNR = SNR_low:SNR_step:SNR_high
            counter_N = counter_N + 1;

            fprintf('Class: %d\n', count_classes);
            fprintf('SNR: %d\n', SNR);

            file = load(append(files_path, postfix_data, num2str(count_classes), '_', num2str(SNR),'.mat'));

            if train_flag == 1
                train_data(counter_N*NTrain+1:(counter_N+1)*NTrain, :) = file.signal_data;
                train_label(counter_N*NTrain+1:(counter_N+1)*NTrain, 1) = count_classes;
            else  
                test_data(counter_N*NTrain+1:(counter_N+1)*NTrain, :) = file.signal_data;
            end
        end 
    end

    if train_flag == 1
        save(append(files_path, postfix_data, '.mat'), postfix_data, '-v7.3');
        save(append(files_path, postfix_label, '.mat'), postfix_label, '-v7.3');
    else
        save(append(files_path, postfix_data, '.mat'), postfix_data, '-v7.3');
    end
end