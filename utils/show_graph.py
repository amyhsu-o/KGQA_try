import argparse
import subprocess
import webbrowser


def show_graph(path: str) -> None:
    # HTTP server
    subprocess.Popen(["python", "-m", "http.server", "8888"])

    # Open the generated graph in the default web browser
    webbrowser.open(f"http://localhost:8888/{path}/graph.html")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str, required=True)
    args = arg_parser.parse_args()

    show_graph(args.path)
