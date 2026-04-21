from db_mode import get_db_mode, set_db_mode
from db_sqlite import (
    setup_database,
    list_persons,
    list_appearances,
    rename_person,
    set_monitored,
    delete_person,
)
from db_vector import delete_embeddings_for_person
from face_processing import ensure_folders, run_webcam


def change_database_mode():
    print("\n===== DATABASE MODE =====")
    print("1. Local")
    print("2. Remote")
    print("3. Cloud")

    choice = input("Choose database mode: ").strip()

    if choice == "1":
        set_db_mode("local")
        print("Database mode set to local.")
    elif choice == "2":
        set_db_mode("remote")
        print("Database mode set to remote.")
    elif choice == "3":
        set_db_mode("cloud")
        print("Database mode set to cloud.")
    else:
        print("Invalid choice.")


def main():
    ensure_folders()
    setup_database()

    while True:
        current_mode = get_db_mode()

        print("\n===== MENU =====")
        print(f"Current database mode: {current_mode}")
        print("1. Run webcam")
        print("2. Show persons")
        print("3. Show appearances")
        print("4. Rename person")
        print("5. Mark person as monitored")
        print("6. Unmark person as monitored")
        print("7. Change database mode")
        print("8. Delete person")
        print("0. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            run_webcam()

        elif choice == "2":
            persons = list_persons()
            if len(persons) == 0:
                print("No persons found.")
            else:
                for person in persons:
                    print(person)

        elif choice == "3":
            appearances = list_appearances()
            if len(appearances) == 0:
                print("No appearances found.")
            else:
                for appearance in appearances:
                    print(appearance)

        elif choice == "4":
            person_id = int(input("Enter person ID: ").strip())
            new_name = input("Enter new name: ").strip()
            rename_person(person_id, new_name)
            print("Person renamed.")

        elif choice == "5":
            person_id = int(input("Enter person ID: ").strip())
            notes = input("Enter notes: ").strip()
            set_monitored(person_id, 1, notes)
            print("Person marked as monitored.")

        elif choice == "6":
            person_id = int(input("Enter person ID: ").strip())
            set_monitored(person_id, 0, "")
            print("Person unmarked as monitored.")

        elif choice == "7":
            change_database_mode()

        elif choice == "8":
            person_id = int(input("Enter person ID to delete: ").strip())
            confirm = input("Are you sure? (y/n): ").strip().lower()

            if confirm == "y":
                delete_person(person_id)
                delete_embeddings_for_person(person_id)
                print("Person deleted successfully.")
            else:
                print("Delete cancelled.")

        elif choice == "0":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()