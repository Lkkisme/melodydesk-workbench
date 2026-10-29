"""Tests application callbacks directly; does not inject desktop input."""
import copy
import unittest
from unittest.mock import patch
from app import App
from music import Project, parse_jianpu


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.app.withdraw()
        self.app.report_callback_exception = lambda *args: None
        notes, _ = parse_jianpu('1:1 1:1 3:1 0:1')
        self.app.set_project(Project(notes=notes))
        self.app.update_idletasks()

    def tearDown(self):
        self.app.destroy()

    def test_edit_preserves_original_timing_and_undo(self):
        self.app.table.selection_set('0')
        self.app.pitch.set('62')
        self.app.note_start.set('0')
        self.app.note_duration.set('0.5')
        self.app.edit_note()
        self.assertEqual(self.app.project.notes[0].pitch, 62)
        self.assertEqual(self.app.project.notes[0].raw_duration, .5)
        self.assertEqual(self.app.project.notes[0].duration, .5)
        self.app.undo()
        self.assertEqual(self.app.project.notes[0].pitch, 60)
        self.assertEqual(self.app.project.notes[0].duration, 1)

    def test_split_then_merge(self):
        self.app.table.selection_set('0')
        self.app.split_note()
        self.assertEqual(len(self.app.project.notes), 4)
        self.app.table.selection_set('0', '1')
        self.app.merge_notes()
        self.assertEqual(len(self.app.project.notes), 3)
        self.assertEqual(self.app.project.notes[0].duration, 1)

    def test_invalid_edit_does_not_mutate_project(self):
        original = copy.deepcopy(self.app.project)
        self.app.table.selection_set('0')
        self.app.pitch.set('62')
        self.app.note_start.set('0')
        self.app.note_duration.set('3')
        with self.assertRaises(ValueError):
            self.app.edit_note()
        self.assertEqual(self.app.project, original)

    def test_manual_input_and_settings(self):
        self.app.tonic.set('D')
        self.app.bpm.set('100')
        self.app.from_manual()
        self.assertEqual(self.app.project.notes[0].pitch, 62)
        self.assertEqual(self.app.project.bpm, 100)
        self.assertFalse(self.app.project.source)

    def test_regrid_is_undoable(self):
        self.app.bpm.set('60')
        with patch('app.messagebox.askyesno', return_value=True):
            self.app.requantize()
        self.assertEqual(self.app.project.notes[1].start, .5)
        self.app.undo()
        self.assertEqual(self.app.project.notes[1].start, 1)

    def test_controls_fit_default_window(self):
        self.app.deiconify()
        self.app.update()
        self.assertGreater(self.app.table.winfo_height(), 80)
        self.assertLess(self.app.progress.winfo_rooty() - self.app.winfo_rooty() + self.app.progress.winfo_height(), self.app.winfo_height())

    def test_organize_and_restore_reference(self):
        original=copy.deepcopy(self.app.project.notes)
        self.app.organize_score()
        self.app.table.selection_set('0')
        self.app.pitch.set('65')
        self.app.edit_note()
        self.app.restore_reference()
        self.assertEqual(self.app.project.notes,original)

    def test_collapse_different_pitches_is_undoable(self):
        original=copy.deepcopy(self.app.project)
        self.app.table.selection_set('1','2')
        self.app.pitch.set('62')
        self.app.collapse_selected()
        self.assertEqual(len(self.app.project.notes),2)
        self.assertEqual(self.app.project.notes[1].pitch,62)
        self.app.undo()
        self.assertEqual(self.app.project,original)

    def test_drag_changes_score_without_changing_reference(self):
        from types import SimpleNamespace
        original=copy.deepcopy(self.app.project.reference_notes)
        self.app.drag=(0,0,0,'move',copy.deepcopy(self.app.project.notes),100,10)
        self.app.release_roll(SimpleNamespace(x=0,y=-20))
        self.assertEqual(self.app.project.notes[0].pitch,62)
        self.assertEqual(self.app.project.reference_notes,original)

    def test_local_cleanup_keeps_other_notes_and_persists_reference(self):
        import tempfile
        from pathlib import Path
        original=copy.deepcopy(self.app.project)
        self.app.table.selection_set('1')
        self.app.organize_score(True)
        self.assertEqual(self.app.project.notes[0],original.notes[0])
        self.assertEqual(self.app.project.notes[2],original.notes[2])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'project.json'
            self.app.project.save(path)
            self.assertEqual(Project.load(path),self.app.project)

    def test_overlapping_drag_is_rejected_without_mutation(self):
        from types import SimpleNamespace
        original=copy.deepcopy(self.app.project)
        self.app.drag=(0,0,0,'end',copy.deepcopy(self.app.project.notes),100,10)
        with self.assertRaises(ValueError):
            self.app.release_roll(SimpleNamespace(x=100,y=0))
        self.assertEqual(self.app.project,original)


if __name__ == '__main__':
    unittest.main()
