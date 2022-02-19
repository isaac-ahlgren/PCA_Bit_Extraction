file = "./new_doyle/doyle_500khz_2ndfloor_ds20.csv";
period_length = 4000;
obs_num = 10;
host_num = 2;
device_nums = [2,3,4];
shift_amount = 250;

M = readmatrix(file);

bit_stats = generate_bits(M, period_length, obs_num, shift_amount, host_num, device_nums);

function bit_stats = generate_bits(data_matrix, period_length, obs_num, shift_amount, host_num, device_nums)
    bit_stats = zeros(length(device_nums),shift_amount);
    host_samples = get_window(data_matrix(:,host_num), 1, period_length*obs_num);
    [host_bits,host_A,host_V,host_T,host_U] = pca_sig(host_samples,period_length);

    for i = 1:length(device_nums)
         device_buffer = data_matrix(:,device_nums(i));

         for shift = 1:shift_amount
             buf = get_window(device_buffer, shift, period_length*obs_num);
             [device_bits,dev_A,dev_V,dev_T,dev_U] = pca_sig(buf,period_length);
             percent = compare_bits(device_bits,host_bits);
             bit_stats(i,shift) = percent;
         end
    end
end

% Used to actually make the graphs
function make_graph(directory, name, x_axis_name, y_axis_name, x_axis, BA1, BA2, BA3, BA4)
    hl_gray50 = [180 180 180]/255;
    hl_orange = [255 147 0]/255;
    hl_gray25= [102 102 102]/255;
    google_blue= [101 157 246]/255;
    
    clf
    hold on
    xlabel(x_axis_name);
    ylabel(y_axis_name);
    plot(x_axis,BA1,'LineWidth',1,'color',hl_gray25);
    plot(x_axis,BA2,'LineWidth',1,'color',hl_orange);
    plot(x_axis,BA3,'LineWidth',1,'color',hl_gray50);
    plot(x_axis,BA4,'LineWidth',1,'color',google_blue);
    legend("Same Power Strip", "Outlets in Same Room", "Outlets in Different Rooms", "Outlets on Different Floors");
    set(gca,'FontSize',16,'FontName', 'Times New Roman');
    set(gca,'YMinorGrid','on');
    set(gca,'LineWidth',0.8);
    xlim([0 length(BA1)]);
    title("Bit Agreement by Sample Shift")
    export_fig ./figures/figure_name.pdf
    figure_name = directory+ "/" + name;
    movefile("./figures/figure_name.pdf", figure_name);
end

function N = get_window(M, sample_start, sample_length)
    N = M(sample_start:sample_start+sample_length);
end

function res = compare_bits(B1,B2)
    agreed = 0;
    for i = 1:length(B1)
        if B1(i) == B2(i)
            agreed = agreed + 1;
        end
    end
    res = agreed / length(B1) * 100;
end