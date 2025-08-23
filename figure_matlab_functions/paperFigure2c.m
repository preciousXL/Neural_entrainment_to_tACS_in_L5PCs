clear all; clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% E-field distribution in layer 5 SOI %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_layer5      = 1;
plot_three_part  = 0;
save_figure_path = fullfile("C:\Users\Snowp\Desktop\paperFigures", "figure2c-1.tif");
%% 1. Load layer 5 surface and E-field distribution
folderPath = "C:\Users\Snowp\PycharmProjects\pythonProject\24_tACS_PLV\data_paper\data_matlab";
fileName   = "L5SOI_PPh_uniformEF_smooth.mat";
dataPath   = fullfile(folderPath, fileName);
data          = load(dataPath);
layer_surface = data.surface;
list_color    = data.list_color;
%% 2. Load precentral gygus, central sulcus, postcentral gyrus surface mesh indices
% load mesh
fileName    = "pre_central_post_indices_original.mat";
dataPath    = fullfile(folderPath, fileName);
dataIndices = load(dataPath);
precentral_indices  = dataIndices.precentral_indices;
central_indices     = dataIndices.central_indices;
postcentral_indices = dataIndices.postcentral_indices;
% load boundary
fileName     = "pre_central_post_border_coordinates.mat";
dataPath     = fullfile(folderPath, fileName);
dataBoundary = load(dataPath);
precentral_boundary  = dataBoundary.border_pre_central;
postcentral_boundary = dataBoundary.border_post_central;
%% 3. Plot figures
if plot_layer5
    fig = figure('Units', 'centimeters', 'Position', [30, 10, 7, 8]);
    patch(layer_surface, 'FaceVertexCData', list_color, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
    surface_central          = layer_surface; 
    surface_central.faces    = surface_central.faces(central_indices, :);
    surface_central.vertices = surface_central.vertices + [0.0, -33.0, 0.0];
    listcolor_central        = list_color(central_indices, :);
    patch(surface_central, 'FaceVertexCData', listcolor_central, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
    hold on;
    plot3( precentral_boundary(:,1),  precentral_boundary(:,2),  precentral_boundary(:,3), "Color", [0.8 0.8 0.8], LineWidth=1, LineStyle="-");
    plot3(postcentral_boundary(:,1), postcentral_boundary(:,2), postcentral_boundary(:,3), "Color", [0.8 0.8 0.8], LineWidth=1, LineStyle="-");
    ax = gca; axis(ax,'equal'); 
    axis(ax, 'off');
    ax.View = [-90 45];
    camlight(ax, 'head')
    exportgraphics(gcf, save_figure_path, 'Resolution', 600)
end
%% 4. Results in three parts (precentral, central, postcentral)
if plot_three_part
    surface_precentral          = layer_surface; 
    surface_precentral.faces    = surface_precentral.faces(precentral_indices, :);
    surface_precentral.vertices = surface_precentral.vertices + [0.0, 20, 0.0];
    listcolor_precentral        = list_color(precentral_indices, :);
    surface_central          = layer_surface; 
    surface_central.faces    = surface_central.faces(central_indices, :);
    surface_central.vertices = surface_central.vertices + [0.0, 0.0, 0.0];
    listcolor_central        = list_color(central_indices, :);
    surface_postcentral          = layer_surface; 
    surface_postcentral.faces    = surface_postcentral.faces(postcentral_indices, :);
    surface_postcentral.vertices = surface_postcentral.vertices + [0.0, -20, 0.0];
    listcolor_postcentral        = list_color(postcentral_indices, :);
    
    fig = figure('Units', 'centimeters', 'Position', [30, 10, 7, 7]);
    patch( surface_precentral, 'FaceVertexCData',  listcolor_precentral, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
    patch(    surface_central, 'FaceVertexCData',     listcolor_central, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
    patch(surface_postcentral, 'FaceVertexCData', listcolor_postcentral, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
    ax = gca; axis(ax,'equal'); 
    axis(ax, 'off');
    ax.View = [-90 45];
    camlight(ax, 'head')

end