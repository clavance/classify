import os
from bs4 import BeautifulSoup

base_dir = "/.../html_raw/"
second_dir = "/.../html_clean/"
directory = os.fsencode(base_dir)

#loop through all files in directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    #store the CELEX ID
    celex = filename.split(":", 1)[0]

    #initialise error variable as False
    error = 0
    corrigendum = 0
    not_english = 0
    no_body = 0

    #store the document title
    with open(base_dir + filename, "r", encoding='latin1') as myfile:
        soup = BeautifulSoup(myfile, 'html.parser')

        #select the Title and reference tag, find the title text
        if (soup.select("div.listNotice h2.rech")[0].text == "Title and reference"):
            title = soup.select("div.listNotice p")[0].text

            #if the title indicates a corrigendum, error = True
            if (i in title for i in ("CORRIGENDUM", "corrigendum", "Corrigendum")):
                error = 1
                corrigendum = 1

        #if there is a document body, save it
        if (soup.find("div", {"class": "texte"})):
            body = soup.find("div", {"class": "texte"}).text

            # if body contains 'no English' error statement, error = True
            if "There is no English version of this document" in body:
                error = 1
                not_english = 1

        #if there is no document body, error = True
        else:
            error = 1
            no_body = 1

        if (error == 1):
            with open(second_dir + "ERROR_" + celex + ".txt", "w", encoding='latin1') as newfile:
                if (corrigendum == 1):
                    newfile.write("CORRIGENDUM" + "\n")

                if (not_english == 1):
                    newfile.write("NOT IN ENGLISH" + "\n")

                if (no_body == 1):
                    newfile.write("NO DOCUMENT TEXT" + "\n")

                newfile.close()

        else:
            with open(second_dir + celex + ".txt", "w", encoding='latin1') as newfile:
                newfile.write("CELEX: " + celex + "\n")
                newfile.write("Title: " + title + "\n")
                newfile.write("Text: " + "\"" + body + "\"")
                newfile.close()