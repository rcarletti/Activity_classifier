%% Four-class classifier

for t_id = [1, 2, 4]
    for s_id = 1:3
        plot_name = sprintf('FCC - Independent sensors (sensor %d, interval %ds)', ...
            s_id, 164/t_id);
        fig_name = sprintf('conf_fcc_ind_s%d_%ds', s_id, 164/t_id);

        save_confusion(fcc_ind{t_id}.results{s_id}.net.targets, ...
            fcc_ind{t_id}.results{s_id}.net.results, plot_name, fig_name);
    end
    
    plot_name = sprintf('FCC - Unified sensors (interval %ds)', 164/t_id);
    fig_name = sprintf('conf_fcc_all_%ds', 164/t_id);

    save_confusion(fcc_all{t_id}.best_net.targets, ...
        fcc_all{t_id}.best_net.results, plot_name, fig_name);
end

%% One-vs-all classifier

for t_id = [1, 2, 4]
    for a_id = 1:4
        for s_id = 1:3
            plot_name = sprintf('OAA - Independent sensors (activity %d, sensor %d, interval %ds)', ...
                a_id, s_id, 164/t_id);
            fig_name = sprintf('conf_oaa_ind_act%d_s%d_%ds', a_id, s_id, 164/t_id);

            save_confusion(onevsall_ind{t_id}{a_id}.results{s_id}.net.targets, ...
                onevsall_ind{t_id}{a_id}.results{s_id}.net.results, plot_name, fig_name);
        end
    
        plot_name = sprintf('OAA - Unified sensors (activity %d, interval %ds)', a_id, 164/t_id);
        fig_name = sprintf('conf_oaa_all_act%d_%ds', a_id, 164/t_id);

        save_confusion(onevsall_all{t_id}{a_id}.best_net.targets, ...
            onevsall_all{t_id}{a_id}.best_net.results, plot_name, fig_name);
    end
end

%% Save confusion mat to file

function save_confusion(tgt, res, plot_name, fig_name)
    plotconfusion(tgt, res, plot_name);
    print(strcat('plots/', fig_name), '-dpng');
end
