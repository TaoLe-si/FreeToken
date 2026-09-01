import faulthandler, sys
def main():
    sys.argv=['launch']+sys.argv[1:]
    from freetoken.server.launch import launch_server
    launch_server()
if __name__ == "__main__":
    main()