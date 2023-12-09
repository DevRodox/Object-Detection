import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.draw import disk, polygon, rectangle
from skimage import measure
class ShapeAnalyzer:
    def __init__(self, colored_images):
        self.colored_images = colored_images
    def calculate_properties(self, prop):
        properties = {
            'area': prop.area,
            'bounding_box': prop.bbox,
            'centroid': prop.centroid,
            'convex_area': prop.convex_area,
            'convex_image': prop.convex_image,
            'coordinates': prop.coords,
            'eccentricity': prop.eccentricity,
            'equivalent_diameter': prop.equivalent_diameter,
            'euler_number': prop.euler_number,
            'extent': prop.extent,
            'filled_area': prop.filled_area,
            'filled_image': prop.filled_image,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length,
            'orientation': prop.orientation,
            'perimeter': prop.perimeter,
            'solidity': prop.solidity,
            'circularity': (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter != 0 else 0,
            'convexity': prop.convex_area / prop.area if prop.area != 0 else 0,
        }

        if hasattr(prop, 'intensity_image'):
            properties['intensity'] = prop.mean_intensity if isinstance(prop.mean_intensity, (float, int)) else prop.median_intensity
        else:
            properties['intensity'] = None

        return properties
    def determine_shape(self, properties):
                num_vertices = len(properties['bounding_box'])

                aspect_ratio_range = (1.5, 2.5)

                minr, minc, maxr, maxc = properties['bounding_box']
                width = maxc - minc
                height = maxr - minr
                aspect_ratio = width / height

                if properties['circularity'] > 0.9 and properties['eccentricity'] < 0.1:
                    return 'Círculo'
                elif properties['convexity'] == 1 and properties['solidity'] < 0.9:
                    return 'Triángulo'
                elif aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]:
                    return 'Rectángulo'
                elif 0.9 <= aspect_ratio <= 1.1:
                    return 'Rectángulo'
                elif num_vertices == 3 or num_vertices == 4: 
                    angles = [0, 0, 0, 0]
                    if width > 0 and height > 0:
                        angles[0] = np.arctan2(height, width) * 180 / np.pi
                        angles[1] = 90 - angles[0]
                        angles[2] = angles[0]
                        angles[3] = angles[1]

                    if sum(angles) == 180:
                        return 'Triángulo'
                    elif sum(angles) == 360:
                        return 'Rectángulo'
    def analyze_shapes(self, colored_image, ax):
                label_image = measure.label(np.any(colored_image > 0, axis=2))
                props = measure.regionprops(label_image)
                ax.imshow(colored_image)
                
                for prop in props:
                    shape_properties = self.calculate_properties(prop)
                    shape_type = self.determine_shape(shape_properties)
                    minr, minc, maxr, maxc = prop.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(minc, minr, shape_type, color='yellow', fontsize=12)
    def make_results_for_all_images(self):
        fig, axes = plt.subplots(1, len(self.colored_images), figsize=(30, 7))
        if len(self.colored_images) == 1:
            axes = [axes]

        category_titles = ['Circles', 'Triangles', 'Rectangles', 'Mezcla de los tres tipos']

        shape_counts = {
            'circles': {'Círculo': 0, 'Triángulo': 0, 'Rectángulo': 0},
            'triangles': {'Círculo': 0, 'Triángulo': 0, 'Rectángulo': 0},
            'rectangles': {'Círculo': 0, 'Triángulo': 0, 'Rectángulo': 0},
            'random_shapes': {'Círculo': 0, 'Triángulo': 0, 'Rectángulo': 0}
        }

        combined_data = []
        for i, colored_image in enumerate(self.colored_images):
            label_image = measure.label(np.any(colored_image > 0, axis=2))
            props = measure.regionprops(label_image)
            ax = axes[i] if len(self.colored_images) > 1 else axes
            ax.imshow(colored_image)
            ax.set_title(category_titles[i])

            current_shape_counts = {'Círculo': 0, 'Triángulo': 0, 'Rectángulo': 0}
            figures_data = []
            for prop in props:
                shape_properties = self.calculate_properties(prop)
                shape_type = self.determine_shape(shape_properties)
                current_shape_counts[shape_type] += 1

                minr, minc, maxr, maxc = prop.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(minc, minr, shape_type, color='yellow', fontsize=12)

                figure_data = {'Category': category_titles[i]}
                figure_data.update(shape_properties)
                figures_data.append(figure_data)

            df_figures = pd.DataFrame(figures_data)
            combined_data.append(df_figures)

            category_names = ['circles', 'triangles', 'rectangles', 'random_shapes']
            if i < len(category_names):
                category_name = category_names[i]
                shape_counts[category_name] = current_shape_counts

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_df = combined_df[combined_df['Category'] != 'Mezcla de los tres tipos']
            combined_df = combined_df[['Category'] + [col for col in combined_df.columns if col != 'Category']]
            combined_df.to_csv('./Propiedades/Combinado.csv', index=False)
        else:
            print("No se pudo crear el archivo combinado porque no se encontraron archivos CSV para las categorías.")

        plt.tight_layout()
        plt.show()

        return shape_counts
