from __future__ import print_function
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools

# If modifying these scopes, delete the file token.json.
#SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly'
SCOPES = 'https://www.googleapis.com/auth/drive'

def create_folder(files, folder_name, parent_folder=None):
    if parent_folder is None:
        metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
    else:
        metadata = {
            'name': folder_name,
            'parents': [parent_folder]
            'mimeType': 'application/vnd.google-apps.folder'
        }
    return files.create(body=metadata, fields='id').execute()


def get_folder(files, folder_name):
    results = files.list(q="name = '" + folder_name + "' and mimeType = 'application/vnd.google-apps.folder'").execute()
    items = results.get('files', [])
    if len(items) == 0:
        return create_folder(files, folder_name)['id']
    else:
        return items[0]['id']

def upload_file(files, folder_id, file_name, file_on_disk):
    media_body = MediaFileUpload(file_on_disk, mimetype='text/plain', resumable=True)
    body = {
      'name': file_name,
      'parents': [folder_id]
      'mimeType': 'text/plain'
    }
    newfile = files.create(body=body, media_body=media_body).execute()

def file_exists(files, folder_id, file_name):
    results = files.list(q="'" + folder_name + "' in parents and name = '" + file_name + "'").execute()
    items = results.get('files', [])
    return len(items) > 0

def get_files_in_folder(files, folder_id):
    results = files.list(q="'" + folder_name + "' in parents").execute()
    items = results.get('files', [])
    return items

def clear_folder(files, folder_id):
    for file in get_files_in_folder(files, folder_id):
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

    print(get_folder(service.files(), "Hej"))
    print(get_folder(service.files(), "HejHej"))
    print(get_folder(service.files(), "HejHejHej"))
    #service.files().create(body="Once upon a time...", media_mime_type="text/plain").execute()

    #for i in range(140):
    FILENAME = "short_file.txt" #"13.json"
    media_body = MediaFileUpload(FILENAME, mimetype='text/plain', resumable=True)
    body = {
      'title': 'My document',
      'description': 'A test document',
      'name': 'file ',
      'mimeType': 'text/plain'
    }
    newfile = service.files().create(body=body, media_body=media_body).execute()
        #print(i)
    print(newfile)

    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print('{0} ({1})'.format(item['name'], item['id']))
            print(service.files().get(fileId=item['id']).execute())

if __name__ == '__main__':
    main()
