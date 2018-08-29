from traceback import format_exception

import googleapiclient
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools

from array import array

import random, time, os, sys
from datetime import datetime


def mean(list):
    return sum(list) / float(len(list))

from drive_data_io import get_folder, get_files_in_folder, create_empty_file, upload_file, download_file, clear_folder, \
    file_exists_by_id, remove_files, get_or_create_folder, delete_file


def print_error():
    # print str(sys.exc_info()[0])
    # print str(sys.exc_info()[1])
    print str(''.join(format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2], None)))
    print


def to_datetime(drive_api_date_string):
    return datetime.strptime(drive_api_date_string, '%Y-%m-%dT%H:%M:%S.%fZ')


class MulticomputerWorker:
    def __init__(self, project_name, main_process=False):
        self.init_files_service()
        self.project_name = project_name
        if main_process:
            self.jobs_folder_id = get_or_create_folder(self.files, project_name + " JOBS")
            self.results_folder_id = get_or_create_folder(self.files, project_name + " RESULTS")
        else:
            self.jobs_folder_id = get_folder(self.files, project_name + " JOBS")
            self.results_folder_id = get_folder(self.files, project_name + " RESULTS")
        self.job_check_interval = 5.0
        self.no_internet_check_interval = 5.0
        self.num_jobs = 0
        self.main_process = main_process

        # State
        self.current_job_n = None
        self.current_job_file_id = None
        self.current_job_progress_file_id = None


    def clear_folders_and_jobs(self):
        while 1:
            try:
                self.num_jobs = 0
                self.jobs_folder_id = clear_folder(self.files, self.jobs_folder_id, self.project_name + " JOBS")
                self.results_folder_id = clear_folder(self.files, self.results_folder_id, self.project_name + " RESULTS")
                print "Cleared folders"
                break
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)


    """
        Waits for and returns the float vector associated with the next free job and marks the job as occupied
    """
    def wait_for_next_job(self):
        while True:
            try:
                next_job = self.find_next_open_job()
                if next_job is not None:
                    return next_job
                else:
                    time.sleep(self.job_check_interval)
            except googleapiclient.errors.HttpError:
                print_error()
                print "Updating folder ids"
                self.jobs_folder_id = get_folder(self.files, self.project_name + " JOBS")
                self.results_folder_id = get_folder(self.files, self.project_name + " RESULTS")
                time.sleep(self.no_internet_check_interval)

    """
        Uploads the result from the current job
    """
    def upload_result(self, score):
        while 1:
            try:
                if file_exists_by_id(self.files, self.current_job_file_id):
                    create_empty_file(self.files, str(self.current_job_n) + "=" + str(float(score)), self.results_folder_id)
                    self.current_job_n = None
                    break
                else:
                    print "Initial job file not found! It seems that the completed job is no longer wanted done, probably because someone else completed it before you. Looking for a new job..."
                    return
            except googleapiclient.errors.HttpError as e:
                if e.resp["status"] == 404:
                    print "googleapiclient.errors.HttpError (404 File not found) in upload_result(). Assuming that the result is already computed. Looking for a new job..."
                    return
                else:
                    print_error()
                    time.sleep(self.no_internet_check_interval)

    def init_files_service(self):
        while 1:
            try:
                SCOPES = 'https://www.googleapis.com/auth/drive'
                store = file.Storage('token.json')
                creds = store.get()
                if not creds or creds.invalid:
                    flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
                    creds = tools.run_flow(flow, store)
                service = build('drive', 'v3', http=creds.authorize(Http()))
                self.service = service
                self.files = self.service.files()
                break
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

    def upload_job(self, np_column_vec):
        job_float_list = [e[0] for e in np_column_vec.tolist()]
        while 1:
            try:
                job_n = self.num_jobs
                temp_file_path = 'tmp/temp.bin'
                if not os.path.exists("tmp/"):
                    os.makedirs("tmp/")
                while 1:
                    try:
                        with open(temp_file_path, 'wb') as output_file:
                            float_array = array('d', job_float_list)
                            float_array.tofile(output_file)
                            break
                    except IOError:
                        print "IOError in upload_job. Waiting 5 seconds and trying again..."
                        time.sleep(5.0)

                upload_file(self.files, str(job_n) + ".bin", "tmp/temp.bin", self.jobs_folder_id)
                try:
                    os.remove(temp_file_path)
                except WindowsError:
                    print "Could not remove temp.bin"
                except OSError:
                    print "Could not remove temp.bin"
                self.num_jobs += 1
                break
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

    def read_job(self, job_file_id):
        result_path = download_file(self.files, job_file_id)
        result = array('d')
        with open(result_path, "rb") as f:
            result.fromstring(f.read())
        try:
            os.remove(result_path)
        except WindowsError:
            print "WindowsError in read_job. Temp file not removed."
        except OSError:
            print "OSError in read_job. Temp file not removed."
        return np.expand_dims(np.array(result), axis=1)

    def get_results(self):
        while 1:
            try:
                # TODO: Fix checking whether all results are done
                result_multiple = [[] for _ in range(self.num_jobs)]
                for file in get_files_in_folder(self.files, self.results_folder_id):
                    splt = file['name'].split("=")
                    if len(splt) == 2:
                        n = int(splt[0])
                        score = float(splt[1])
                        # result[n] = score
                        result_multiple[n].append(score)
                missing = []
                for i in range(self.num_jobs):
                    if len(result_multiple[i]) == 0:
                        missing.append(i)
                if len(missing) == 0:
                    result = map(mean, result_multiple)
                    return None, result
                else:
                    return missing, None
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

    def remove_progress_files(self, job_indices):
        try:
            file_names = map(lambda x: str(x) + " IN_PROGRESS", job_indices)
            remove_files(self.service, file_names, self.jobs_folder_id)
        except googleapiclient.errors.HttpError:
            print_error()
            time.sleep(self.no_internet_check_interval)

    def find_next_open_job(self):
        while 1:
            try:
                if not self.main_process:
                    print "Updating folder ids"
                    self.jobs_folder_id = get_folder(self.files, self.project_name + " JOBS")
                    self.results_folder_id = get_folder(self.files, self.project_name + " RESULTS")
                    while self.jobs_folder_id is None or self.results_folder_id is None:
                        time.sleep(self.job_check_interval)
                        self.jobs_folder_id = get_folder(self.files, self.project_name + " JOBS")
                        self.results_folder_id = get_folder(self.files, self.project_name + " RESULTS")

                candidates = set()
                locked = set()
                file_id = {}
                for file in get_files_in_folder(self.files, self.jobs_folder_id):
                    if len(file['name']) >=4 and file['name'][-4:]==".bin":
                        try:
                            n = int(file['name'][:-4])
                            candidates.add(n)
                            file_id[n] = file['id']
                        except ValueError:
                            print "Unexpected file name in jobs folder (job file)"
                    else:
                        splt = file['name'].split(" ")
                        if len(splt) == 2:
                            try:
                                if splt[1] == "IN_PROGRESS":
                                    locked.add(int(splt[0]))
                            except ValueError:
                                print "Unexpected file name in jobs folder (progress file)"
                possible_jobs = list(candidates - locked)
                if len(possible_jobs) > 0:
                    job_n = random.choice(possible_jobs)
                    job_file_id = file_id[job_n]

                    self.current_job_progress_file_id = create_empty_file(self.files, str(job_n) + " IN_PROGRESS", self.jobs_folder_id)

                    self.current_job_n = job_n
                    self.current_job_file_id = job_file_id
                    return self.read_job(job_file_id)
                else:
                    return None
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

    def is_superfluous(self):
        # print "STARTING IS_SUPERFLUOUS"
        try:
            own_job_file_exists = False
            own_progress_file = None
            competing_progress_files = []
            for file in get_files_in_folder(self.files, self.jobs_folder_id):
                if len(file['name']) >=4 and file['name'][-4:]==".bin":
                    if file['id'] == self.current_job_file_id:
                        own_job_file_exists = True
                else:
                    splt = file['name'].split(" ")
                    if len(splt) == 2:
                        try:
                            if splt[1] == "IN_PROGRESS":
                                if int(splt[0]) == self.current_job_n:
                                    if file['id'] == self.current_job_progress_file_id:
                                        own_progress_file = file
                                    else:
                                        competing_progress_files.append(file)
                        except ValueError:
                            print "Unexpected file name in jobs folder (progress file) in cancel_if_superfluous"
            if not own_job_file_exists:
                print "The job has been cancelled or completed by someone else. Looking for a new job..."
                return True
            elif own_progress_file is None:
                # print "NO OWN PROGRESS FILE"
                return False
            elif len(competing_progress_files) == 0:
                # print "NO COMPETING PROGRESS FILES"
                return False
            else:
                # print "COMPARING DATE TO COMPETING PROGRESS FILES"
                own_date = to_datetime(own_progress_file['createdTime'])
                competing_date = min(map(lambda f: to_datetime(f['createdTime']), competing_progress_files))
                if competing_date < own_date:
                    try:
                        delete_file(self.files, self.current_job_progress_file_id)
                        print "The job was already taken by someone else. Progress file removed. Looking for a new job..."
                    except googleapiclient.errors.HttpError:
                        print "HttpError when deleting progress file in is_superfluous"
                    return True
                else:
                    # print "STARTED BEFORE OF ALL COMPETITORS. CONTINUING EVALUATION... "
                    return False
        except googleapiclient.errors.HttpError:
            print_error()
            time.sleep(self.no_internet_check_interval)
        return False

if __name__ == "__main__":
    process_id = 0
    w = MulticomputerWorker("TestProj2")
    if process_id == 0:
        w.clear_folders_and_jobs()
    def work():
        while True:
            if process_id == 0:
                job = w.find_next_open_job()
            else:
                job = w.wait_for_next_job()
            if job is None:
                print str(process_id) + ": Finished"
                return
            else:
                w.upload_result(sum(job))
                print "Calculated result " + str(sum(job))

    if process_id == 0:
        jobs = [[1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2], [1,2],[4,5],[10,1],[11,2]]
        for vec in jobs:
            w.upload_job(vec)
            print "Uploaded job " + str(vec)
        work()
        print w.get_results()
    else:
        work()