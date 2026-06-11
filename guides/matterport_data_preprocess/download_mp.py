#!/usr/bin/env python3
# Downloads MP public data release
# Run with ./download_mp.py or python download_mp.py on Windows
# -*- coding: utf-8 -*-

import argparse
import os
import re
import socket
import tempfile
import time
import urllib.error
import urllib.request

BASE_URL = 'http://kaldir.vc.cit.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'

FILETYPES = [
    'cameras',
    'matterport_camera_intrinsics',
    'matterport_camera_poses',
    'matterport_color_images',
    'matterport_depth_images',
    'matterport_hdr_images',
    'matterport_mesh',
    'matterport_skybox_images',
    'undistorted_camera_parameters',
    'undistorted_color_images',
    'undistorted_depth_images',
    'undistorted_normal_images',
    'house_segmentations',
    'region_segmentations',
    'image_overlap_data',
    'poisson_meshes',
    'sens'
]

TASK_FILES = {
    'keypoint_matching_data': ['keypoint_matching/data.zip'],
    'keypoint_matching_models': ['keypoint_matching/models.zip'],
    'surface_normal_data': ['surface_normal/data_list.zip'],
    'surface_normal_models': ['surface_normal/models.zip'],
    'region_classification_data': ['region_classification/data.zip'],
    'region_classification_models': ['region_classification/models.zip'],
    'semantic_voxel_label_data': ['semantic_voxel_label/data.zip'],
    'semantic_voxel_label_models': ['semantic_voxel_label/models.zip'],
    'minos': ['mp3d_minos.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'habitat': ['mp3d_habitat.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
    'igibson': ['mp3d_for_igibson.zip'],
    'mp360': [
        'mp3d_360/data_00.zip',
        'mp3d_360/data_01.zip',
        'mp3d_360/data_02.zip',
        'mp3d_360/data_03.zip',
        'mp3d_360/data_04.zip',
        'mp3d_360/data_05.zip',
        'mp3d_360/data_06.zip'
    ]
}


def format_bytes(num):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024:
            return f'{num:.1f}{unit}'
        num /= 1024
    return f'{num:.1f}PB'


def parse_content_range_total(value):
    """
    Parses headers like:
      Content-Range: bytes 100-999/1000
    Returns 1000 or None.
    """
    if not value:
        return None

    match = re.search(r'/(\d+)$', value)
    if not match:
        return None

    return int(match.group(1))


def get_remote_file_info(url, timeout):
    """
    Returns:
      total_size: int or None
      accepts_ranges: bool or None

    None for accepts_ranges means unknown.
    """
    req = urllib.request.Request(
        url,
        method='HEAD',
        headers={'User-Agent': 'mp-download-script/robust'}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_length = resp.headers.get('Content-Length')
            accept_ranges = resp.headers.get('Accept-Ranges', '').lower()

            total_size = int(content_length) if content_length and content_length.isdigit() else None
            accepts_ranges = 'bytes' in accept_ranges if accept_ranges else None

            return total_size, accepts_ranges

    except urllib.error.HTTPError as e:
        # Some servers do not support HEAD. Fall back to unknown.
        if e.code in (403, 405, 501):
            return None, None
        raise

    except urllib.error.URLError:
        return None, None

    except socket.timeout:
        return None, None


def get_release_scans(release_file, timeout, retries):
    last_error = None

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(release_file, timeout=timeout) as scan_lines:
                scans = []
                for scan_line in scan_lines:
                    scan_id = scan_line.decode('utf-8').rstrip('\n')
                    if scan_id:
                        scans.append(scan_id)
                return scans

        except Exception as e:
            last_error = e
            if attempt >= retries:
                break

            sleep_time = min(60, 2 ** attempt)
            print(f'WARNING: failed to fetch scan list: {e}')
            print(f'Retrying in {sleep_time}s...')
            time.sleep(sleep_time)

    raise RuntimeError(f'Failed to fetch release scan list after retries: {last_error}')


def download_release(release_scans, out_dir, file_types, retries, timeout, chunk_size):
    print('Downloading MP release to ' + out_dir + '...')

    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types, retries, timeout, chunk_size)

    print('Downloaded MP release.')


def download_file(url, out_file, retries=10, timeout=60, chunk_size=1024 * 1024):
    """
    Robust downloader.

    Features:
    - Resume from out_file.part if present.
    - Timeout stalled reads.
    - Retry transient errors.
    - Atomic final rename after completion.
    - Size verification when possible.
    """
    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    part_file = out_file + '.part'

    try:
        total_size, accepts_ranges = get_remote_file_info(url, timeout)
    except Exception as e:
        print(f'WARNING: could not query file info for {url}: {e}')
        total_size, accepts_ranges = None, None

    # If final file already exists, verify size if possible.
    if os.path.isfile(out_file):
        local_size = os.path.getsize(out_file)

        if total_size is None:
            print('WARNING: skipping existing file, remote size unknown: ' + out_file)
            return

        if local_size == total_size:
            print('WARNING: skipping existing complete file ' + out_file)
            return

        print(
            f'WARNING: existing file has wrong size: {out_file} '
            f'local={format_bytes(local_size)} remote={format_bytes(total_size)}'
        )

        # If the existing file is smaller, convert it into a partial file for resume.
        # If larger, remove and start clean.
        if local_size < total_size:
            print('Treating existing incomplete file as partial download.')
            os.replace(out_file, part_file)
        else:
            print('Existing file is larger than remote file. Removing and redownloading.')
            os.remove(out_file)

    # If part file already complete, finalize it.
    if os.path.isfile(part_file) and total_size is not None:
        part_size = os.path.getsize(part_file)

        if part_size == total_size:
            print('Completing previously finished partial file: ' + out_file)
            os.replace(part_file, out_file)
            return

        if part_size > total_size:
            print('Partial file is larger than remote file. Removing partial file.')
            os.remove(part_file)

    print('\t' + url + ' > ' + out_file)

    last_error = None

    for attempt in range(retries + 1):
        resume_from = os.path.getsize(part_file) if os.path.isfile(part_file) else 0

        headers = {
            'User-Agent': 'mp-download-script/robust'
        }

        if resume_from > 0:
            headers['Range'] = f'bytes={resume_from}-'

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.getcode()

                # If we asked to resume but server ignored Range and sent a full file,
                # restart the partial file from zero.
                if resume_from > 0 and status == 200:
                    print('Server did not resume download. Restarting this file from zero.')
                    resume_from = 0
                    open_mode = 'wb'

                elif resume_from > 0 and status == 206:
                    print(f'Resuming from {format_bytes(resume_from)}')
                    open_mode = 'ab'

                else:
                    open_mode = 'wb'

                response_total = parse_content_range_total(resp.headers.get('Content-Range'))
                if response_total is not None:
                    total_size = response_total

                # If HEAD failed, infer size from a full GET response when possible.
                if total_size is None and status == 200:
                    content_length = resp.headers.get('Content-Length')
                    if content_length and content_length.isdigit():
                        total_size = int(content_length)

                downloaded = resume_from if open_mode == 'ab' else 0
                last_report = time.monotonic()
                start_time = time.monotonic()

                with open(part_file, open_mode + '') as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.monotonic()
                        if now - last_report >= 5:
                            elapsed = max(now - start_time, 0.001)
                            speed = max(downloaded - resume_from, 0) / elapsed

                            if total_size:
                                percent = downloaded * 100.0 / total_size
                                print(
                                    f'\t  {format_bytes(downloaded)} / {format_bytes(total_size)} '
                                    f'({percent:.1f}%) at {format_bytes(speed)}/s'
                                )
                            else:
                                print(
                                    f'\t  {format_bytes(downloaded)} '
                                    f'at {format_bytes(speed)}/s'
                                )

                            last_report = now

                final_part_size = os.path.getsize(part_file)

                if total_size is not None and final_part_size != total_size:
                    raise IOError(
                        f'incomplete download: got {final_part_size} bytes, '
                        f'expected {total_size} bytes'
                    )

                os.replace(part_file, out_file)
                print('\tDone: ' + out_file)
                return

        except urllib.error.HTTPError as e:
            last_error = e

            # 416 means requested range is not satisfiable.
            # This can happen if the .part file is already complete.
            if e.code == 416:
                if os.path.isfile(part_file) and total_size is not None:
                    part_size = os.path.getsize(part_file)

                    if part_size == total_size:
                        print('Partial file was already complete. Finalizing.')
                        os.replace(part_file, out_file)
                        return

                print('Server rejected resume range. Removing partial file and restarting.')
                if os.path.isfile(part_file):
                    os.remove(part_file)

            else:
                print(f'WARNING: HTTP error while downloading {url}: {e}')

        except (
            urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            ConnectionError,
            IOError,
            OSError
        ) as e:
            last_error = e
            print(f'WARNING: download interrupted for {url}: {e}')

        if attempt >= retries:
            break

        sleep_time = min(60, 2 ** attempt)
        print(f'Retrying in {sleep_time}s... attempt {attempt + 1}/{retries}')
        time.sleep(sleep_time)

    raise RuntimeError(f'Failed to download after {retries} retries: {url}\nLast error: {last_error}')


def download_scan(scan_id, out_dir, file_types, retries, timeout, chunk_size):
    print('Downloading MP scan ' + scan_id + ' ...')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for ft in file_types:
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + ft + '.zip'
        out_file = os.path.join(out_dir, ft + '.zip')
        download_file(url, out_file, retries=retries, timeout=timeout, chunk_size=chunk_size)

    print('Downloaded scan ' + scan_id)


def download_task_data(task_data, out_dir, retries, timeout, chunk_size):
    print('Downloading MP task data for ' + str(task_data) + ' ...')

    for task_data_id in task_data:
        if task_data_id in TASK_FILES:
            files = TASK_FILES[task_data_id]

            for filepart in files:
                url = BASE_URL + RELEASE_TASKS + filepart
                localpath = os.path.join(out_dir, filepart)
                localdir = os.path.dirname(localpath)

                if not os.path.isdir(localdir):
                    os.makedirs(localdir, exist_ok=True)

                download_file(url, localpath, retries=retries, timeout=timeout, chunk_size=chunk_size)
                print('Downloaded task data ' + task_data_id)


def main():
    parser = argparse.ArgumentParser(
        description=
        '''
        Downloads MP public data release.

        Example invocation:
          python download_mp.py -o base_dir --id ALL --type matterport_mesh

        The -o argument is required and specifies the base_dir local directory.

        After download:
          base_dir/v1/scans is populated with scan data
          base_dir/v1/tasks is populated with task data

        Unzip scan files from:
          base_dir/v1/scans

        Unzip task files from:
          base_dir/v1/tasks/task_name

        The --type argument is optional.
        All data types are downloaded if unspecified.

        The --id ALL argument downloads all house data.
        Use --id house_id to download one house.

        The --task_data argument is optional and downloads task data/model files.
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-o',
        '--out_dir',
        required=True,
        help='directory in which to download'
    )

    parser.add_argument(
        '--task_data',
        default=[],
        nargs='+',
        help='task data files to download. Any of: ' + ','.join(TASK_FILES.keys())
    )

    parser.add_argument(
        '--id',
        default='ALL',
        help='specific scan id to download or ALL to download entire dataset'
    )

    parser.add_argument(
        '--type',
        nargs='+',
        help='specific file types to download. Any of: ' + ','.join(FILETYPES)
    )

    parser.add_argument(
        '--retries',
        type=int,
        default=10,
        help='number of retries per file after interrupted downloads. Default: 10'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='socket timeout in seconds. If no data arrives for this long, retry. Default: 60'
    )

    parser.add_argument(
        '--chunk_size',
        type=int,
        default=1024 * 1024,
        help='download chunk size in bytes. Default: 1048576'
    )

    args = parser.parse_args()

    print('By pressing ENTER to continue you confirm that you have agreed to the MP terms of use as described at:')
    print(TOS_URL)
    print('***')
    input('Press ENTER to continue, or CTRL-C to exit.')

    release_file = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file, timeout=args.timeout, retries=args.retries)
    file_types = FILETYPES

    # Download task data.
    if args.task_data:
        invalid_task_ids = set(args.task_data) - set(TASK_FILES.keys())

        if invalid_task_ids:
            print('ERROR: Unrecognized task data id: ' + str(sorted(invalid_task_ids)))
            return

        out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
        download_task_data(
            args.task_data,
            out_dir,
            retries=args.retries,
            timeout=args.timeout,
            chunk_size=args.chunk_size
        )

        print('Done downloading task_data for ' + str(args.task_data))
        input('Press ENTER to continue on to main dataset download, or CTRL-C to exit.')

    # Download specific file types?
    if args.type:
        invalid_types = set(args.type) - set(FILETYPES)

        if invalid_types:
            print('ERROR: Invalid file type: ' + str(sorted(invalid_types)))
            return

        file_types = args.type

    if args.id and args.id != 'ALL' and args.id != 'all':
        scan_id = args.id

        if scan_id not in release_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
            return

        out_dir = os.path.join(args.out_dir, RELEASE, scan_id)
        download_scan(
            scan_id,
            out_dir,
            file_types,
            retries=args.retries,
            timeout=args.timeout,
            chunk_size=args.chunk_size
        )

    elif 'minos' not in args.task_data and (args.id == 'ALL' or args.id == 'all'):
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire MP release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all MP scans of type ' + ','.join(file_types))

        print('Existing complete files will be skipped.')
        print('Incomplete downloads are saved as .part files and resumed automatically.')
        print('***')
        input('Press ENTER to continue, or CTRL-C to exit.')

        out_dir = os.path.join(args.out_dir, RELEASE)
        download_release(
            release_scans,
            out_dir,
            file_types,
            retries=args.retries,
            timeout=args.timeout,
            chunk_size=args.chunk_size
        )


if __name__ == "__main__":
    main()

