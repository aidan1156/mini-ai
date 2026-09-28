-- A reply's parent must be a top-level message (no replies to replies).
CREATE TRIGGER chat_messages_parent_is_root_insert
BEFORE INSERT ON chat_messages
WHEN NEW.parent_id IS NOT NULL
 AND (SELECT parent_id FROM chat_messages WHERE id = NEW.parent_id) IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'parent_id must point to a top-level message');
END;

CREATE TRIGGER chat_messages_parent_is_root_update
BEFORE UPDATE OF parent_id ON chat_messages
WHEN NEW.parent_id IS NOT NULL
 AND (
     NEW.parent_id = NEW.id
     OR (SELECT parent_id FROM chat_messages WHERE id = NEW.parent_id) IS NOT NULL
     -- a message that already has replies can't become a reply itself
     OR EXISTS (SELECT 1 FROM chat_messages WHERE parent_id = NEW.id)
 )
BEGIN
    SELECT RAISE(ABORT, 'parent_id must point to a top-level message');
END;
