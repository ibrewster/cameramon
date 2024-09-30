import configparser
import logging
import os

import requests
from shapely.geometry import Polygon
from shapely.prepared import prep
from shapely.ops import unary_union

logger = logging.getLogger("cameramon")


def process_coordinates(coords):
    points = coords.split()
    points = [(float(y), float(z)) for y, z in [x.split(',') for x in points]]
    poly = Polygon(points)
    return poly


def load_zones(monitor):
    conf_ini = configparser.ConfigParser()
    MODULE_PATH = os.path.dirname(__file__)
    conf_ini.read(os.path.join(MODULE_PATH, 'cameramon.ini'))
    
    # Get a list of zones for our monitor
    logger.info(f"Loading zones for monitor {monitor}")
    api_user = conf_ini['general']['zm_api_user']
    api_password = conf_ini['general']['zm_api_password']
    auth = {'user': api_user,
            'pass': api_password, }
    zm_url = conf_ini['general']['zm_portal_url']
    api_url = f"{zm_url}/api/zones/forMonitor/{monitor}.json"
    resp = requests.get(api_url, auth)
    zone_list = resp.json()
    zone_list = zone_list['zones']
    zone_coords = [process_coordinates(x['Zone']['Coords']) for x in
                   zone_list if x['Zone']['Type'] == 'Active']

    logger.debug(str(zone_coords))
    zones = prep(unary_union(zone_coords))
    logger.debug(str(zones))
    
    no_car_names = ['Yard', 'Front Bush1', 'Side Bushes', 'Porch']
    car_zone_coords = [
        process_coordinates(x['Zone']['Coords'])
        for x in zone_list
        if x['Zone']['Type'] == 'Active'
        and x['Zone']['Name'] not in no_car_names]
    
    car_zone = unary_union(car_zone_coords)
    logger.debug(str(car_zone))
    car_zones = prep(car_zone)
    
    return (zones, car_zones)