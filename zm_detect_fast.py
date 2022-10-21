#!/usr/bin/python3

import requests
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('eventid', help='event ID to retrieve')
    ap.add_argument('monitorid', help='monitor id - needed for mask', 
                    default = None, nargs = "?")
    ap.add_argument('monitorname', help='monitor name',
                    default = '', nargs = '?')
    ap.add_argument('reason', help='reason for event (notes field in ZM)',
                    default = '', nargs = '?')
    ap.add_argument('eventpath',
                    help='path to store object image file',
                    default='', nargs = '?')
    
    args, _ = ap.parse_known_args()
    args = vars(args)
    
    if not args.get('eventid'):
        print ('--eventid required')
        exit(1)
        
    query_args = {'eventid': args.get('eventid'),
                  'monitorid': args.get('monitorid'),
                  'eventpath': args.get('eventpath')
                  }
    
    resp = requests.get('http://127.0.0.1:5003/detect', query_args)
    print(resp.text)
    
    if 'detected' in resp.text:
        exit(0)
    else:
        exit(1)