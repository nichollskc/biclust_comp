function []=runMultiCluster(tensor_file, output_files, params, N_file)

disp(params)
Ncomp = str2num(params.K_init)

n_ind = str2num(fileread(N_file))

M = readmatrix(tensor_file);

[n_rows, n_cols] = size(M);

n_gene = n_cols;
n_cell = n_rows / n_ind;

disp('Dimensions of matrix are (genes, individuals, cells):')
disp([n_gene, n_ind, n_cell])

Tensor = reshape(M', [n_gene, n_ind, n_cell]);

% perform tensor decomposition with semi-nonnegative constraint in the Z-mode
[output_vector_X,output_vector_Y,output_vector_Z,output_value]=MultiCluster(Tensor,Ncomp);

nonzero_counts = zeros(Ncomp, 3);
for index = 1:Ncomp
    nonzero_counts(index,1) = sum(output_vector_X(:,index)~=0);
    nonzero_counts(index,2) = sum(output_vector_Y(:,index)~=0);
    nonzero_counts(index,3) = sum(output_vector_Z(:,index)~=0);
end
disp('Counts of non-zero elements in X (genes), Y (individuals) and Z (cells)')
disp(nonzero_counts)

writematrix(output_vector_X, output_files.B, 'Delimiter','tab');
writematrix(output_vector_Y, output_files.A, 'Delimiter','tab');
writematrix(output_vector_Z, output_files.Z, 'Delimiter','tab');

writematrix(Ncomp, output_files.K, 'Delimiter','tab');

end
