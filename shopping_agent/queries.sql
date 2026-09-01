-- database: ./store.db



ALTER TABLE Orders DROP COLUMN ordered_at;

CREATE TRIGGER IF NOT EXISTS set_order_timestamp
AFTER INSERT ON Orders
FOR EACH ROW
WHEN NEW.ordered_at IS NULL
BEGIN
    UPDATE Orders 
    SET ordered_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;
