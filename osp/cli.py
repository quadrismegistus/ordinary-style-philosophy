"""
Unified CLI for the Ordinary Style Philosophy project.

Usage:
    osp check                        # validate raw data files
    osp pipeline                     # run full pipeline
    osp pipeline parse feats         # run specific steps
    osp pipeline parse --limit 100   # with options
    osp export                       # export derived data
    osp export -o data/release/      # to specific directory
    osp dashboard                    # launch Streamlit app
"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("""usage: osp <command> [args...]

commands:
  check       Validate that raw data files are in place
  pipeline    Run the data pipeline (assemble, slice, parse, feats, classify)
  export      Export derived data for publication
  dashboard   Launch the Streamlit dashboard

Run 'osp <command> --help' for command-specific options.""")
        sys.exit(0)

    command = sys.argv[1]
    # Remove the subcommand from argv so the sub-CLI sees the right args
    sys.argv = [f"osp {command}"] + sys.argv[2:]

    if command == "check":
        from .check_data import main as check_main
        check_main()
    elif command == "pipeline":
        from .pipeline import main as pipeline_main
        pipeline_main()
    elif command == "export":
        from .export import main as export_main
        export_main()
    elif command == "dashboard":
        _run_dashboard()
    else:
        print(f"Unknown command: {command}")
        print("Run 'osp --help' for available commands.")
        sys.exit(1)


def _run_dashboard():
    import os
    import subprocess
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard")
    app_path = os.path.join(dashboard_dir, "app.py")
    if not os.path.exists(app_path):
        print(f"Dashboard not found at {app_path}")
        sys.exit(1)
    # Pass through any extra args (e.g. --server.port 8502)
    cmd = ["streamlit", "run", app_path] + sys.argv[1:]
    os.chdir(dashboard_dir)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
