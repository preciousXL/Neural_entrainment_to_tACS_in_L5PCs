clear all; clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NonUniform PPh and its difference to unifrom %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
folderPath = "C:\Users\Snowp\PycharmProjects\pythonProject\24_tACS_PLV\data_paper\data_matlab";
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
%% 1. PLV of nonuniform E-field
save_figure_path = fullfile("C:\Users\Snowp\Desktop\paperFigures", "figure6c-1.tif");
fileName   = "L5SOI_polarization_nonUniformEF_smooth.mat";
dataPath   = fullfile(folderPath, fileName);
data          = load(dataPath);
layer_surface = data.surface;
list_color    = data.list_color;
% plot figures
fig = figure('Units', 'centimeters', 'Position', [30, 10, 7, 7]);
patch(layer_surface, 'FaceVertexCData', list_color, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
surface_central          = layer_surface; 
surface_central.faces    = surface_central.faces(central_indices, :);
surface_central.vertices = surface_central.vertices + [0.0, -33.0, 0.0];
listcolor_central        = list_color(central_indices, :);
patch(surface_central, 'FaceVertexCData', listcolor_central, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
hold on;
ax = gca; axis(ax,'equal'); 
axis(ax, 'off');
ax.View = [-90 45];
camlight(ax, 'head')
exportgraphics(gcf, save_figure_path, 'Resolution', 600)

%% 2. PLV: absolute error —— (uniform - nonuniform)
save_figure_path = fullfile("C:\Users\Snowp\Desktop\paperFigures", "figure6c-2.tif");
fileName   = "L5SOI_abosolute_error_of_polarization_between_uniform_and_nonUniform_smooth.mat";
dataPath   = fullfile(folderPath, fileName);
data          = load(dataPath);
layer_surface = data.surface;
list_color    = data.list_color;
% plot figures
fig = figure('Units', 'centimeters', 'Position', [30, 10, 7, 7]);
patch(layer_surface, 'FaceVertexCData', list_color, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
surface_central          = layer_surface; 
surface_central.faces    = surface_central.faces(central_indices, :);
surface_central.vertices = surface_central.vertices + [0.0, -33.0, 0.0];
listcolor_central        = list_color(central_indices, :);
patch(surface_central, 'FaceVertexCData', listcolor_central, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
hold on;
ax = gca; axis(ax,'equal'); 
axis(ax, 'off');
ax.View = [-90 45];
camlight(ax, 'head')
exportgraphics(gcf, save_figure_path, 'Resolution', 600)

%% 3. PLV: relative error —— 100*(uniform - nonuniform)/nonuniform
save_figure_path = fullfile("C:\Users\Snowp\Desktop\paperFigures", "figure6c-3.tif");
fileName   = "L5SOI_relative_error_of_polarization_between_uniform_and_nonUniform_smooth.mat";
dataPath   = fullfile(folderPath, fileName);
data          = load(dataPath);
layer_surface = data.surface;
list_color    = data.list_color;
% plot figures
fig = figure('Units', 'centimeters', 'Position', [30, 10, 7, 7]);
patch(layer_surface, 'FaceVertexCData', list_color, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
surface_central          = layer_surface; 
surface_central.faces    = surface_central.faces(central_indices, :);
surface_central.vertices = surface_central.vertices + [0.0, -33.0, 0.0];
listcolor_central        = list_color(central_indices, :);
patch(surface_central, 'FaceVertexCData', listcolor_central, 'FaceColor', 'flat','EdgeColor', 'none', 'FaceLighting','gouraud');
hold on;
ax = gca; axis(ax,'equal'); 
axis(ax, 'off');
ax.View = [-90 45];
camlight(ax, 'head')
exportgraphics(gcf, save_figure_path, 'Resolution', 600)

