import os
import cv2
import subprocess
import numpy as np
import argparse
from princeton365.utils.utils_io import load_config_from_yaml


def generate_charuco_boards(args):
    boards = []
    for index in range(args.num_boards):
        board = create_board(index, args)
        boards.append(board)
    return boards

def create_board(index, args):
    width_px = int(args.paper_size_in[0] * args.ppi) # width in pixels
    height_px = int(args.paper_size_in[1] * args.ppi) # height in pixels
    square_px = int(args.square_length * 39.37 * args.ppi) # square length in pixels
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    
    marker_ids = np.arange(
            index * args.total_markers,
            index * args.total_markers + args.total_markers,
        )
    if args.board_type == "grid":
        board = cv2.aruco.GridBoard(
            size = (args.cols, args.rows),
            markerLength = args.square_length,
            markerSeparation = args.marker_separation,
            dictionary = dictionary,
            ids = marker_ids
        )

        marker_separation_px = int(args.marker_separation * 39.37 * args.ppi) # marker separation in pixels
        margin = int((width_px - (square_px + marker_separation_px) * (args.cols - 1) - marker_separation_px) / 2)
        img = cv2.aruco.drawPlanarBoard(board, (width_px, height_px), marginSize=margin, borderBits=1)

    else:
        board = cv2.aruco.CharucoBoard(
            (args.cols, args.rows),
            args.square_length,
            args.marker_length,
            dictionary,
            ids = marker_ids
        )
        margin_px_x = int((width_px - square_px * args.cols) / 2)
        img = cv2.aruco.CharucoBoard.generateImage(board, (width_px, height_px), marginSize=margin_px_x)

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"board_{index}.png")
        cv2.imwrite(output_path, img)
        pdf_output_path = output_path.replace(".png", ".pdf")
        try:
            subprocess.run(["/u/sa6924/.local/bin/img2pdf", "-s", f"{args.ppi}dpi", "--output", pdf_output_path, output_path], check=True)
            print(f"Saved PDF: {pdf_output_path}")
        except subprocess.CalledProcessError:
            print("PDF conversion failed")

    return board
            

def main():
    parser = argparse.ArgumentParser(description="Generate and optionally save or view Charuco/Grid boards")
    
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--board_type", type=str, default="grid", help="Either 'charuco' or 'grid'")
    

    cli_args = parser.parse_args()

    args = load_config_from_yaml(cli_args.board_type, cli_args.config)
    generate_charuco_boards(args)

if __name__ == "__main__":
    main()

