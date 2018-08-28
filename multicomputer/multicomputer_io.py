from traceback import format_exception

import googleapiclient
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools

from array import array

import random, time, os, sys

from drive_data_io import get_folder, get_files_in_folder, create_empty_file, upload_file, download_file, clear_folder, \
    file_exists_by_id


def print_error():
    # print str(sys.exc_info()[0])
    # print str(sys.exc_info()[1])
    print str(''.join(format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2], None)))
    print

class MulticomputerWorker:
    def __init__(self, project_name):
        self.init_files_service()
        self.project_name = project_name
        self.jobs_folder_id = get_folder(self.files, project_name + " JOBS")
        self.results_folder_id = get_folder(self.files, project_name + " RESULTS")
        self.job_check_interval = 5.0
        self.no_internet_check_interval = 5.0
        self.num_jobs = 0

        # State
        self.current_job_n = None
        self.current_job_file_id = None


    def clear_folders_and_jobs(self):
        while 1:
            try:
                self.num_jobs = 0
                clear_folder(self.service, self.jobs_folder_id)
                clear_folder(self.service, self.results_folder_id)
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
                time.sleep(self.no_internet_check_interval)

    """
        Uploads the result from the current job
    """
    def upload_result(self, score):
        while 1:
            try:
                if file_exists_by_id(self.files, self.current_job_file_id):
                    create_empty_file(self.files, str(self.current_job_n) + "=" + str(score), self.results_folder_id)
                    self.current_job_n = None
                    break
                else:
                    print "Initial job file not found! It seems that the completed job is no longer wanted done, probably because someone else completed it before you. Looking for a new job..."
                    return
            except googleapiclient.errors.HttpError:
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

    def upload_job(self, float_list):
        while 1:
            try:
                job_n = self.num_jobs
                temp_file_path = 'tmp/temp.bin'
                if not os.path.exists("tmp/"):
                    os.makedirs("tmp/")
                with open(temp_file_path, 'wb') as output_file:
                    float_array = array('d', float_list)
                    float_array.tofile(output_file)
                upload_file(self.files, str(job_n) + ".bin", "tmp/temp.bin", self.jobs_folder_id)
                os.remove(temp_file_path)
                self.num_jobs += 1
                break
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

    def read_job(self, job_file_id):
        while 1:
            try:
                result_path = download_file(self.files, job_file_id)
                result = array('d')
                with open(result_path) as f:
                    result.fromstring(f.read())
                try:
                    os.remove(result_path)
                except WindowsError:
                    print "WindowsError in read_job. Temp file not removed."
                return result
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

    def get_results(self):
        while 1:
            try:
                result = [0] * self.num_jobs
                for file in get_files_in_folder(self.files, self.results_folder_id):
                    splt = file['name'].split("=")
                    if len(splt) == 2:
                        n = int(splt[0])
                        score = float(splt[1])
                        result[n] = score
                return result
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

    def find_next_open_job(self):
        while 1:
            try:
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

                    create_empty_file(self.files, str(job_n) + " IN_PROGRESS", self.jobs_folder_id)

                    self.current_job_n = job_n
                    self.current_job_file_id = job_file_id
                    return self.read_job(job_file_id)
                else:
                    return None
            except googleapiclient.errors.HttpError:
                print_error()
                time.sleep(self.no_internet_check_interval)

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