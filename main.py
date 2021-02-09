import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import datetime
import numpy as np

def grid_of_region():
    # Coordinates of selected region
    xmin, ymin, xmax, ymax = [144.92972075, -37.78972559, 144.93566736, -37.79490412]

    num_cells = 10
    x_step = (xmax - xmin) / num_cells
    y_step = abs((ymax - ymin) / num_cells)

    #grid = list()
    df = pd.DataFrame(columns = ['cell', 'geometry'])

    for j in range(0, 10):
        for i in range(0, 10):
            count = j * 10 + i + 1
            x0 = xmin + x_step * i
            x1 = xmin + x_step * (i + 1)
            y0 = ymin - y_step * j
            y1 = ymin - y_step * (j + 1)
            df = df.append({'cell' : count, 'geometry': Polygon([[x0,y0], [x1, y0], [x1,y1], [x0, y1]])}, ignore_index=True)

    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    return gdf

def split_into_cells(grid):
    road_df = pd.read_csv("Road_Corridor.csv")
    road_df['the_geom'] = road_df['the_geom'].apply(wkt.loads)

    traffic_df = pd.read_csv("Traffic_Count_Vehicle_Classification_2014-2017.csv")
    traffic_df['date'] = pd.to_datetime(traffic_df['date'] + ' ' + traffic_df['time'])
    traffic_df['average_speed'] = traffic_df['average_speed'].fillna(0)

    col_list = list(traffic_df.columns[7:22])
    traffic_df["total_count"] = traffic_df[col_list].sum(axis=1)
    traffic_df.drop(col_list, "columns", inplace=True)

    traffic_df = traffic_df.merge(road_df, left_on="road_segment", right_on="SegID")

    traffic_df['geometry'] = traffic_df['the_geom']
    traffic_df.drop("the_geom", "columns", inplace=True)

    gdf = gpd.GeoDataFrame(traffic_df, geometry='geometry')

    coords = [144.9275, -37.7896, 144.9359, -37.7967]
    region_gdf = gdf.cx[144.9275:144.9359, -37.7896:-37.7967]

    date_time_lwr = datetime.datetime(2016, 1, 1)
    date_time_upr = datetime.datetime(2016, 12, 31)

    mask = (region_gdf['date'] > date_time_lwr) & (region_gdf['date'] <= date_time_upr)

    region_gdf = region_gdf.loc[mask]

    region_gdf.set_index('date', inplace=True)
    region_gdf = region_gdf.between_time("7:00", "18:00")
    region_gdf.reset_index(inplace=True)

    cols = ['date', 'average_speed', "total_count", "geometry", "road_segment"]

    region_gdf = region_gdf[cols]
    road_segs = list(set(region_gdf.road_segment))
    road_seg_df = pd.DataFrame(columns=cols)
    sorted_region_gdf = gpd.GeoDataFrame(columns = cols)

    days = list()

    for i in road_segs:
        road_seg_df = road_seg_df.append(region_gdf[region_gdf.road_segment == i][-1:])
        sorted_region_gdf = sorted_region_gdf.append(region_gdf[region_gdf.road_segment == i][-48:])

    #ax = grid.plot(color = "white", edgecolor = "black")
    #region_4d_gdf.plot(ax=ax)
    #plt.show()

    def grid_search(geom):
        coords = list()
        for i in range(100):
            if grid.geometry[i].intersects(geom):
               coords.append(i + 1)
        return coords

    road_seg_df['cells'] = road_seg_df["geometry"].apply(grid_search)

    # 4 days of readings 12 hours each day
    cell_speeds = np.zeros((100, 48))
    travel_demands = [0] * 48

    for i, seg in enumerate(road_segs):
        for j in range(48):
            cells = road_seg_df.cells.iloc[i]
            avg_speed = sorted_region_gdf[sorted_region_gdf.road_segment == seg].average_speed.iloc[j]
            travel_demands[j] += sorted_region_gdf[sorted_region_gdf.road_segment == seg].total_count.iloc[j]
            for cell in cells:
                cell_speeds[cell - 1][j] = avg_speed

    region_travel_demands = np.zeros((48, 4))

    for i, demand in enumerate(travel_demands):
        region_travel_demands[i][2] = demand
        region_travel_demands[i][3] = (i % 12) + 1

    np.savetxt("melb_label_region.csv", region_travel_demands, delimiter=",")

    avg_speed_imgs = list()

    for i in range(48):
        avg_speed_imgs.append(cell_speeds[:, i].reshape(10,10))

    np.savetxt("melb_speed_region.csv", np.concatenate(avg_speed_imgs, axis=0), delimiter = ',')


    daily_speeds = list()

    # Somewhat messy way of calculating pearson coeff - very slow
    for i in range(12):
        daily_speed = np.corrcoef(cell_speeds[:, [i , i + 12, i + 24, i + 36]], rowvar=True)
        daily_speed[np.isnan(daily_speed)] = 0
        daily_speed = np.where(daily_speed < 0.47, 0, daily_speed)
        row_sums = daily_speed.sum(axis=1)
        daily_speed = daily_speed / row_sums[:, np.newaxis]
        daily_speed[np.isnan(daily_speed)] = 0
        daily_speeds.append(daily_speed)

    # 4 days of readings
    adj = np.concatenate([np.concatenate(daily_speeds, axis=0)] * 4, axis=0)

    np.savetxt("melb_speed_region_adjacency_0.47.csv", adj, delimiter=",")


if __name__ == '__main__':
    split_into_cells(grid_of_region())