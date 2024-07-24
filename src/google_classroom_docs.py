import io
import pickle
import re
import os
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from dotenv import load_dotenv
load_dotenv()


class GoogleClassroom :
    def __init__(self, credentials) -> None :
        # Authenticate the connection and load the credentials
        self.credentials = credentials
        # Load the classroom service
        self.classroom_service = build('classroom', 'v1', credentials=self.credentials)
    
    
    def list_courses(self):
        """Fetch and return a list of all courses.
        """
        courses = []
        page_token = None

        while True:
            response = self.classroom_service.courses().list(pageToken=page_token).execute()
            courses.extend(response.get('courses', []))
            page_token = response.get('nextPageToken', None)
            if not page_token:
                break

        return courses
    
    
    def list_announcements(self, course_id):
        """Fetch and return a list of announcements for a given course.
        """
        announcements = []
        page_token = None

        while True:
            response = self.classroom_service.courses().announcements().list(
                courseId=course_id, 
                pageToken=page_token
                ).execute()
            
            announcements.extend(response.get('announcements', []))
            page_token = response.get('nextPageToken', None)
            if not page_token:
                break

        return announcements
    
    
    def list_materials(self, course_id):
        """Fetch and return a list of materials for a given course.
        """
        materials = []
        page_token = None

        while True:
            response = self.classroom_service.courses().courseWorkMaterials().list(
                courseId=course_id, 
                pageToken=page_token
                ).execute()

            materials.extend(response.get('courseWorkMaterial', []))
            page_token = response.get('nextPageToken', None)
            if not page_token:
                break

        return materials
    
    

class GoogleDrive :
    def __init__(self, credentials) -> None :
        self.credentials = credentials
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        
    
    def list_drive_files(self, material):
        """Fetch and return a list of drive files for a given material.
        """
        drive_files = []

        for item in material.get('materials', []):
            if 'driveFile' in item:
                drive_files.append(item['driveFile']['driveFile'])

        return drive_files
    
    
    def download_drive_file(self, service, file_id, file_name):
        """Download a file from Google Drive.
        """
        print(f"downloading file : {file_name}")
        request = service.files().get_media(fileId=file_id)
        file_io = io.FileIO(file_name, 'wb')
        downloader = MediaIoBaseDownload(file_io, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {file_name} {int(status.progress() * 100)}%.")
            
    
    def filter_drive_files(
        self,
        drive_file, 
        supported_formats=("ppt", "pptx", "docx", "pdf")
        ) :
        """Filter drive files based on their extension
        """
        if "title" in drive_file :
            return drive_file["title"].split(".")[-1].lower() in supported_formats
        else :
            return False
    
    

class GoogleServices :
    SCOPES = ["https://www.googleapis.com/auth/classroom.courses.readonly",
          "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
          "https://www.googleapis.com/auth/classroom.announcements.readonly",
          "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
          "https://www.googleapis.com/auth/drive.readonly",]

    def __init__(self) -> None:
        self.creds = self.authenticate()
    
    
    def authenticate(self) :
        creds = None
        
        # If token already exists, simply load it
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                print("credentials loaded")
                creds = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                print("verification initiated")
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', 
                    self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                print("credentials saved")
                pickle.dump(creds, token)

        return creds
    
    
    def extract_drive_files_in_classroom(self, classroom, drive) :
        print("Starting File Extraction from Google Classroom...")
        documents = []
        courses = classroom.list_courses()

        for course in courses:
            # Get all the announcements for the course
            announcements = classroom.list_announcements(
                course['id']
            )
            
            # Add 
            for announcement in announcements:
                drive_files = drive.list_drive_files(announcement)
                drive_files = list(filter(drive.filter_drive_files, drive_files))
                for drive_file in drive_files[:5]:
                    documents.append(
                        {
                            "title" : drive_file["title"],
                            "id" : drive_file["id"],
                            "type" : {
                                "type": "announcement",
                                "title" : announcement["text"]
                            },
                            "course" : course["name"],
                        }
                    )

            materials = classroom.list_materials(course['id'])
            for material in materials:
                drive_files = drive.list_drive_files(material)
                drive_files = list(filter(drive.filter_drive_files, drive_files))
                for drive_file in drive_files :
                    documents.append(
                        {
                            "title" : drive_file["title"],
                            "id" : drive_file["id"],
                            "type" : {
                                "type": "material",
                                "title" : material["title"]
                            },
                            "course" : course["name"],
                        }
                    )
                    
        print("Done with Extraction...")
        return documents
    
    
    def download_drive_files_from_classroom(self, docs, drive) :
        """Download all files listed in the list of documents from google drive
        into their respective course folders
        """
        print("Downloading Drive Files...")
        if not os.path.exists("data") :
            os.mkdir("data")

        for document in docs :
            try :
                document["course"] = document["course"].replace("/", "_")
                
                if not os.path.exists(f"data/{document['course']}") :
                    os.mkdir(f"data/{document['course']}")
                drive.download_drive_file(
                    drive.drive_service, document['id'], 
                    f"data/{document['course']}/{document['title']}"
                )
            except Exception as exp :
                print(exp)
        print("Done Downloading Drive Files...")

    
    def match_substring_case_insensitive(title, announcement, substring="module"):
        """Return True if the substring is found in the title or the announcement
        Not part of core logic. Used to create a set of sub-ducuments for testing
        """
        pattern = re.compile(re.escape(substring), re.IGNORECASE)
        title_match = pattern.search(title)
        announcement_match = pattern.search(announcement)

        return (title_match is not None) or (announcement_match is not None)
    
    


if __name__ == "__main__" :
    google_services = GoogleServices()
    
    drive = GoogleDrive(google_services.creds)
    classroom = GoogleClassroom(google_services.creds)
    
    # Get all drive files in the classroom
    docs = google_services.extract_drive_files_in_classroom(classroom, drive)
    # Download the drive files
    google_services.download_drive_files_from_classroom(docs, drive)
    
    # # Testing
    # module_docs = list(filter(
    #     lambda doc: google_services.match_substring_case_insensitive(
    #         doc["title"], 
    #         doc["type"]["title"], 
    #         "module"
    #         ),
    #     docs)
    #     )
    
    # google_services.download_drive_files_from_classroom(module_docs, drive)