from __future__ import print_function

import os

import googleapiclient
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from httplib2 import Http
from oauth2client import file, client, tools
import io, time
from array import array

import datetime

# If modifying these scopes, delete the file token.json.
#SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly'
SCOPES = 'https://www.googleapis.com/auth/drive'
DRIVE_PAGE_SIZE = 1000

def create_folder(files, folder_name, parent_folder=None):
    if parent_folder is None:
        metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
    else:
        metadata = {
            'name': folder_name,
            'parents': [parent_folder],
            'mimeType': 'application/vnd.google-apps.folder'
        }
    return files.create(body=metadata, fields='id').execute()



def get_folder(files, folder_name):
    results = files.list(pageSize=DRIVE_PAGE_SIZE, q="name = '" + folder_name + "' and mimeType = 'application/vnd.google-apps.folder'").execute()
    items = results.get('files', [])
    if len(items) == 0:
        return create_folder(files, folder_name)['id']
    else:
        return items[0]['id']

def create_empty_file(files, file_name, parent_folder=None):
    if parent_folder is None:
        metadata = {
            'name': file_name,
        }
    else:
        metadata = {
            'name': file_name,
            'parents': [parent_folder],
        }
    return files.create(body=metadata, fields='id').execute()


def upload_file(files, file_name, file_on_disk, folder_id=None):
    media_body = MediaFileUpload(file_on_disk, mimetype='text/plain', resumable=True)
    if folder_id is None:
        body = {
          'name': file_name,
          'mimeType': 'text/plain'
        }
    else:
        body = {
          'name': file_name,
          'parents': [folder_id],
          'mimeType': 'text/plain'
        }

    newfile = files.create(body=body, media_body=media_body).execute()

def download_file(files, file_id):
    request = files.get_media(fileId=file_id)
    result_path = "tmp/" + file_id + " " + str(time.time())#datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    if not os.path.exists("tmp/"):
        os.makedirs("tmp/")
    fh = io.FileIO(result_path, mode='w')#io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
    return result_path


def file_exists(files, folder_id, file_name):
    results = files.list(pageSize=DRIVE_PAGE_SIZE, q="'" + folder_id + "' in parents and name = '" + file_name + "'").execute()
    items = results.get('files', [])
    return len(items) > 0

def file_exists_by_id(files, file_id):
    try:
        items = files.get(fileId=file_id).execute()#files.list(q="id = '" + file_id + "'").execute()
        # items = results.get('files', [])
        return len(items) > 0
    except googleapiclient.errors.HttpError:
        return False

def get_files_in_folder(files, folder_id):
    results = files.list(pageSize=DRIVE_PAGE_SIZE, q="'" + folder_id + "' in parents").execute()
    items = results.get('files', [])
    return items

def clear_folder(service, folder_id):
    def delete_file(request_id, response, exception):
        if exception is not None:
            print("Exception in delete_file batch: " + str(exception))
        else:
            pass
    batch = service.new_batch_http_request(callback=delete_file)
    for file in get_files_in_folder(service.files(), folder_id):
        batch.add(service.files().delete(fileId=file['id']))
    batch.execute()


def remove_files(service, file_names, folder_id):
    def delete_file(request_id, response, exception):
        if exception is not None:
            print("Exception in delete_file batch: " + str(exception))
        else:
            pass

    batch = service.new_batch_http_request(callback=delete_file)
    for file in get_files_in_folder(service.files(), folder_id):
        if file['name'] in file_names:
            batch.add(service.files().delete(fileId=file['id']))
            print("Removing " + file['name'])
    batch.execute()

def delete_file(files, file_id):
    files.delete(fileId=file_id).execute()

def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))

    # Call the Drive v3 API

    # from array import array
    # output_file = open('array.bin', 'wb')
    # float_array = array('d', [3.14, 2.7, 0.0, -1.0, 1.1])
    # float_array.tofile(output_file)
    # output_file.close()
    # upload_file(service.files(), "array.bin", "array.bin", get_folder(service.files(), "Hej"))

    print(get_folder(service.files(), "Hej"))
    print(get_folder(service.files(), "HejHej"))
    print(get_folder(service.files(), "HejHejHej"))

    print(file_exists_by_id(service.files(), get_folder(service.files(), "Hej")))
    print(file_exists_by_id(service.files(), "abcd"))
    print()

    create_empty_file(service.files(), "hehehe.txt")
    create_empty_file(service.files(), "hahaha.txt", get_folder(service.files(), "HejHejHej"))

    #service.files().create(body="Once upon a time...", media_mime_type="text/plain").execute()

    #for i in range(140):
    # FILENAME = "short_file.txt" #"13.json"
    # media_body = MediaFileUpload(FILENAME, mimetype='text/plain', resumable=True)
    # body = {
    #   'title': 'My document',
    #   'description': 'A test document',
    #   'name': 'file ',
    #   'mimeType': 'text/plain'
    # }
    # newfile = service.files().create(body=body, media_body=media_body).execute()
    #     #print(i)
    # print(newfile)

    results = service.files().list(
        pageSize=None, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print('{0} ({1})'.format(item['name'], item['id']))
            # print(service.files().get(fileId=item['id']).execute())
            if item['name'] == "array.bin":
                download_file(service.files(), item['id'])


if __name__ == '__main__':
    main()
