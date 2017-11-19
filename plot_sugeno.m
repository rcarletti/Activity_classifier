function plot_sugeno(sugeno)
% PLOT_SUGENO Plot Sugeno training and validation outputs, and errors.
    figure;

    subplot(2,2,1);
    plot(1:length(sugeno.training_data(:,5)), sugeno.training_data(:,5), '*r', ...
         1:length(sugeno.training_out(:,1)), sugeno.training_out(:,1), 'ob');
    legend('Training Data', 'ANFIS Output');

    subplot(2,2,2);
    plot(1:length(sugeno.validation_data(:,5)), sugeno.validation_data(:,5), '*r', ...
         1:length(sugeno.validation_out(:,1)), sugeno.validation_out(:,1), 'ob');
    legend('Validation Data', 'ANFIS Output');

    subplot(2,2,[3,4]);
    plot(1:length(sugeno.train_err), sugeno.train_err, 'r', ...
         1:length(sugeno.check_err), sugeno.check_err, 'b');
    legend('Training error', 'Validation error');
end
