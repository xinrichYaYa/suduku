
/**
 * 页面坐标转换成canvas的本地坐标
 * @param {MouseEvent} event
 * @returns {{x:number,y:number}}
 */
function relMouseCoords(event) {
  var totalOffsetX = 0;
  var totalOffsetY = 0;
  var canvasX = 0;
  var canvasY = 0;
  var currentElement = this;

  do {
    totalOffsetX += currentElement.offsetLeft;
    totalOffsetY += currentElement.offsetTop;
  } while ((currentElement = currentElement.offsetParent));
  canvasX = event.pageX - totalOffsetX;
  canvasY = event.pageY - totalOffsetY;

  return {
    x: canvasX,
    y: canvasY,
  };
}
HTMLCanvasElement.prototype.relMouseCoords = relMouseCoords;

// Javascript doesn't have 'contains' so added here for later readability
// Array.prototype.contains = function (element) {
//   for (var i = 0; i < this.length; i++) {
//     if (this[i] == element) {
//       return true;
//     }
//   }
//   return false;
// };

var SquareSize = 3;
var BoardSize = SquareSize * SquareSize;

function AllowedValues(n) {
  this._mask = n;
}

AllowedValues.prototype.getSingle = function () {
  // Count number of on bits from 1..9
  var single = 0;
  var count = 0;
  for (var i = 1; i <= BoardSize; i++)
    if ((this._mask & (1 << i)) != 0) {
      count++;
      single = i;
    }
  return count == 1 ? single : 0;
};

// Used when the answer is known at the Cell, level this sets the only allowed value to be that answer
AllowedValues.prototype.setSingle = function (n) {
  this._mask = 1 << n;
};

AllowedValues.prototype.count = function () {
  // Count number of on bits from 1..9
  var count = 0;
  for (var i = 1; i <= BoardSize; i++)
    if ((this._mask & (1 << i)) != 0) count++;
  return count;
};

AllowedValues.prototype.isAllowed = function (n) {
  return n >= 1 && n <= BoardSize && (this._mask & (1 << n)) != 0;
};

AllowedValues.prototype.removeValues = function (bm) {
  this._mask &= ~bm._mask;
};

AllowedValues.prototype.allowedValuesArray = function () {
  var ret = new Array();
  for (var i = 1; i <= BoardSize; i++)
    if (((1 << i) & this._mask) != 0) ret.push(i);
  return ret;
};

AllowedValues.prototype.clone = function () {
  return new AllowedValues(this._mask);
};

// Add / remove a single value to the bitmask (used for user notes)
AllowedValues.prototype.add = function (n) {
  if (n >= 1 && n <= BoardSize) this._mask |= 1 << n;
};

AllowedValues.prototype.remove = function (n) {
  if (n >= 1 && n <= BoardSize) this._mask &= ~(1 << n);
};

AllowedValues.prototype.isEmpty = function () {
  return this.count() == 0;
};

function Cell(value) {
  this._value = value; // 0 means unassigned
  this._allowed = new AllowedValues(0x3e); // all possible
  this._answer = 0; // no answer
  /**
   * 是否为预置值（例如数独固定数字）
   * @type {boolean}
   */
  this._given = false;
  /**
   * 用户笔记（手动候选数字）
   * @type {AllowedValues}
   */
  this._notes = new AllowedValues(0); // user pencil notes (manual candidates)
}

Cell.prototype.clone = function (value) {
  var clone = new Cell();
  clone._value = this._value;
  clone._allowed = this._allowed.clone();
  clone._answer = this._answer;
  clone._given = this._given;
  clone._notes = this._notes.clone();
  return clone;
};

Cell.prototype.single = function (value) {
  this._value = value; // value user (or auto solve functions) has assigned as a possible answer
  this._allowed = new AllowedValues(0x3e); // the allowed values as a bit mask
  this._answer = 0; // calculated as the only possible correct value
};

Cell.prototype.valueMask = function () {
  return this._value == 0 ? 0 : 1 << this._value;
};

Cell.prototype.hasAnswer = function () {
  return this._answer != 0;
};

Cell.prototype.getAnswer = function () {
  return this._answer;
};

Cell.prototype.setAnswer = function (n) {
  if (n < 0 || n > 9) throw "Illegal value not in the range 1..9.";
  this._answer = n;
};

Cell.prototype.getValue = function () {
  return this._value;
};

Cell.prototype.setValue = function (n) {
  if (n < 0 || n > 9) throw "Illegal value not in the range 1..9.";
  if (n != 0 && !this._allowed.isAllowed(n)) throw "Not allowed.";
  this._value = n;
  this._given = false;
  // Clear any manual notes when a value is set
  this._notes = new AllowedValues(0);
};

/**
 * 设置该单元格为预置值（不可修改）
 * @param {number} n - 要设置的预置值
 */
Cell.prototype.setGiven = function (n) {
  if (n < 0 || n > 9) throw "Illegal value not in the range 1..9.";
  this._value = n;
  this._given = n != 0;
  this._answer = 0;
  // Clear notes for givens
  this._notes = new AllowedValues(0);
};

/**
 * 判断该单元格是否是预置值（不可修改）
 * @returns {boolean} 是否为预置状态
 */
Cell.prototype.isGiven = function () {
  return this._given;
};

Cell.prototype.isAssigned = function () {
  return this._value != 0;
};

Cell.prototype.clear = function () {
  this._value = 0; // means unassigned
  this._allowed = new AllowedValues(0x3e); // all possible
  this._answer = 0;
  // this._given = 0;
  this._given = false;
  this._notes = new AllowedValues(0);
};

// Toggle a user note for this cell. If the note exists remove it, otherwise add it.
Cell.prototype.toggleNote = function (n) {
  if (n < 1 || n > BoardSize) return;
  if (this._notes.isAllowed(n)) this._notes.remove(n);
  else this._notes.add(n);
};

Cell.prototype.getNotesArray = function () {
  return this._notes.allowedValuesArray();
};

Cell.prototype.isAllowed = function (value) {
  return this._allowed.isAllowed(value);
};

Cell.prototype.setAllowed = function (value) {
  this._allowed = new AllowedValues(value);
};

Cell.prototype.getAllowedClone = function (value) {
  return this._allowed.clone();
};

var SibType = {
  Row: 1,
  Col: 2,
  Square: 3,
};

function CellLocation(row, col) {
  this.row = row;
  this.col = col;
}

CellLocation.empty = new CellLocation(-1, -1);

CellLocation.prototype.isEmpty = function () {
  return this.row < 0;
};

CellLocation.prototype.modulo = function (n) {
  if (n < 0) return n + BoardSize;
  return n % BoardSize;
};

CellLocation.prototype.left = function () {
  return new CellLocation(this.row, this.modulo(this.col - 1));
};

CellLocation.prototype.right = function () {
  return new CellLocation(this.row, this.modulo(this.col + 1));
};

CellLocation.prototype.up = function () {
  return new CellLocation(this.modulo(this.row - 1), this.col);
};

CellLocation.prototype.down = function () {
  return new CellLocation(this.modulo(this.row + 1), this.col);
};

CellLocation.prototype.toString = function () {
  return "Row=" + String(this.row) + "Col=" + String(this.col);
};

CellLocation.prototype.getSquare = function () {
  return 3 * Math.floor(this.row / 3) + Math.floor(this.col / 3);
};

CellLocation.prototype.equals = function (a) {
  return a.row == this.row && a.col == this.col;
};

CellLocation.prototype.notEquals = function (a) {
  return a.row != this.row || a.col != this.col;
};

// Enumerator for CellLocations of all cells
CellLocation.grid = function () {
  var locs = new Array();
  for (var i = 0; i < BoardSize; i++)
    for (var j = 0; j < BoardSize; j++) locs.push(new CellLocation(i, j));
  return locs;
};

// Enumerator for CellLocations of cell siblings in the same row
CellLocation.prototype.rowSibs = function () {
  var locs = new Array();
  for (var i = 0; i < BoardSize; i++)
    if (i != this.col) locs.push(new CellLocation(this.row, i));
  return locs;
};

// Enumerator for CellLocations of cell siblings in the same column
CellLocation.prototype.colSibs = function () {
  var locs = new Array();
  for (var i = 0; i < BoardSize; i++)
    if (i != this.row) locs.push(new CellLocation(i, this.col));
  return locs;
};

// Enumerator for CellLocations of cell siblings in the same square
CellLocation.prototype.squareSibs = function () {
  var locs = new Array();
  var baseRow = 3 * Math.floor(this.row / 3); // this is how to convert float to an "int" - Javascript doesn't have ints!
  var baseCol = 3 * Math.floor(this.col / 3);
  for (var i = 0; i < SquareSize; i++) {
    var r = baseRow + i;
    for (var j = 0; j < SquareSize; j++) {
      var c = baseCol + j;
      if (r != this.row || c != this.col) locs.push(new CellLocation(r, c));
    }
  }
  return locs;
};

CellLocation.prototype.getSibs = function (type) {
  switch (type) {
    case SibType.Row:
      return this.rowSibs();
    case SibType.Col:
      return this.colSibs();
    case SibType.Square:
      return this.squareSibs();
  }
};

/**
 * 数独棋盘类
 */
class Board {
  constructor() {
    function MultiDimArray(rows, cols) {
      var a = new Array(rows);
      for (var i = 0; i < rows; i++) {
        a[i] = new Array(cols);
        for (var j = 0; j < cols; j++) a[i][j] = new Cell();
      }
      return a;
    }

    this._digits = MultiDimArray(BoardSize, BoardSize);
    this._isSolved = false;
    this._isValid = false;
  }
}

Board.prototype.clone = function () {
  var clone = new Board();
  clone._isSolved = this._isSolved;
  clone._isValid = this._isValid;
  clone._digits = new Array(BoardSize);
  for (var i = 0; i < BoardSize; i++) {
    clone._digits[i] = new Array(BoardSize);
    for (var j = 0; j < BoardSize; j++)
      clone._digits[i][j] = this._digits[i][j].clone();
  }
  return clone;
};

Board.prototype.copyTo = function (target) {
  target._isSolved = this._isSolved;
  target._isValid = this._isValid;
  for (var i = 0; i < BoardSize; i++)
    for (var j = 0; j < BoardSize; j++)
      target._digits[i][j] = this._digits[i][j].clone();
};

Board.prototype.getCell = function (loc) {
  return this._digits[loc.row][loc.col];
};

Board.prototype.setCell = function (loc, value) {
  this._digits[loc.row][loc.col] = value;
};

Board.prototype.clear = function () {
  for (var i = 0; i < BoardSize; i++)
    for (var j = 0; j < BoardSize; j++) this._digits[i][j].clear();
  this.updateAllowed();
};

Board.prototype.reset = function () {
  // return Baord to only the givens
  for (var i = 0; i < BoardSize; i++)
    for (var j = 0; j < BoardSize; j++) {
      var cell = this._digits[i][j];
      if (!cell.isGiven()) cell.clear();
    }
  this.updateAllowed();
};

Board.prototype.checkIsValidSibs = function (loc, digit, locs) {
  for (var i = 0; i < locs.length; i++) {
    var loc = locs[i];
    var cell = this._digits[loc.row][loc.col];
    if (cell.getAnswer() == digit) return false;
  }
  return true;
};

Board.prototype.checkIsValid = function (loc, digit) {
  // Checks if the digit can go in that CellLocation by checking it doesn't
  // exist in either the row, col or square siblings
  if (!this.checkIsValidSibs(loc, digit, loc.colSibs())) return false;
  if (!this.checkIsValidSibs(loc, digit, loc.rowSibs())) return false;
  if (!this.checkIsValidSibs(loc, digit, loc.squareSibs())) return false;

  return true;
};

Board.prototype.acceptPossibles = function () {
  var more = false;
  var locs = CellLocation.grid();
  for (var i = 0; i < locs.length; i++) {
    var loc = locs[i];
    var cell = this._digits[loc.row][loc.col];
    if (
      !cell.isAssigned() &&
      cell.hasAnswer() &&
      this.checkIsValid(loc, cell.getAnswer())
    ) {
      cell.setValue(cell.getAnswer()); // if unassigned and has the answer then assign the answer
      more = true;
    }
  }
  return more;
};

Board.prototype.checkForHiddenSingles = function (loc, st) {
  // Check each cell - if not assigned and has no answer then check its siblings
  // get all its allowed then remove all the allowed
  var cell = this.getCell(loc);
  if (!cell.isAssigned() && !cell.hasAnswer()) {
    var allowed = cell.getAllowedClone(); // copy of bit mask of allowed values for this cell
    var locs = loc.getSibs(st);
    for (var i = 0; i < locs.length; i++) {
      var sib = locs[i];
      var sibCell = this.getCell(sib);
      if (!sibCell.isAssigned())
        allowed.removeValues(sibCell.getAllowedClone()); // remove allowed values from siblings
    }
    var answer = allowed.getSingle(); // if there is only one allowed value left (i.e. this cell is the only one amonsgt its sibs with this allowed value)
    // then apply it as the answer. Note getSingle will return 0 (i.e. no anser) if the number of allowed values is not exactly one
    if (answer != 0) {
      cell.setAnswer(answer);
      return true; // no need to check others sibling collections
    }
  }
  return false;
};

Board.prototype.findCellWithFewestChoices = function () {
  var minCellLocation = CellLocation.empty;
  var minCount = 9;
  var locs = CellLocation.grid();
  for (var i = 0; i < locs.length; i++) {
    var loc = locs[i];
    var cell = this.getCell(loc);
    if (!cell.isAssigned()) {
      var count = cell.getAllowedClone().count();
      if (count < minCount) {
        minCellLocation = loc;
        minCount = count;
      }
    }
  }
  return minCellLocation;
};

Board.prototype.updateAllowed = function () {
  // Called whenever the user sets a value or via auto solve
  // Updates the allowed values for each cell based on existing digits
  // entered in a cell's row, col or square
  var cols = new Array(BoardSize);
  var rows = new Array(BoardSize);
  var squares = new Array(BoardSize);

  // First aggregate assigned values to rows, cols, squares
  var locs = CellLocation.grid();
  for (var i = 0; i < locs.length; i++) {
    var loc = locs[i];
    // Disallow for all cells in this row
    var cons = this.getCell(loc).valueMask();
    rows[loc.row] |= cons;//contains
    cols[loc.col] |= cons;
    squares[loc.getSquare()] |= cons;
  }

  // For each cell, aggregate the values already set in that row, col and square.
  // Since the aggregate is a bitmask, the bitwise inverse of that is therefore the allowed values.
  this._isValid = true;
  this._isSolved = true;
  for (var i = 0; i < locs.length; i++) {
    var loc = locs[i];
    // Set allowed values
    var contains = rows[loc.row] | cols[loc.col] | squares[loc.getSquare()];
    var cell = this.getCell(loc);
    cell.setAllowed(~contains); // set allowed values to what values are not already set in this row, col or square
    cell.setAnswer(0); //clear any previous answers
    // As an extra step look for "naked singles", i.e. cells that have only one allowed value, and use
    // that to set the answer (note this is different from the "value" as this can only be assigned
    // by the user or any auto solve functions like "accept singles"
    if (!cell.isAssigned()) {
      this._isSolved = false;
      var mask = new AllowedValues(~contains);
      var count = mask.count();
      if (count == 0) this._isValid = false;
      else if (count == 1) cell.setAnswer(mask.getSingle());
    }
  }

  // Step 2: Look for "hidden singles".
  // For each row, col, square, count number of times each digit appears.
  // If any appear once then set that as the answer for that cell.
  // Count in rows
  for (var i = 0; i < locs.length; i++) {
    var loc = locs[i];
    if (!this.checkForHiddenSingles(loc, SibType.Row))
      if (!this.checkForHiddenSingles(loc, SibType.Col))
        // first check row sibs for a hiddne single
        // then check cols
        this.checkForHiddenSingles(loc, SibType.Square); // then check square
  }

  // TO DO: Add code here to detect naked/hidden doubles/triples/quads
};

Board.prototype.trySolve = function (loc, value) {
  // empty CellLocation allowed
  if (!loc.isEmpty()) {
    // assign a value to a CellLocation if provided
    var cell = this.getCell(loc);
    if (!cell.isAllowed(value)) throw "Internal error.";
    cell.setValue(value);
  }

  do {
    this.updateAllowed();
    if (!this._isValid) return false;
  } while (this.acceptPossibles()); // keep doing deterministic answers

  if (this._isSolved) return true;

  if (!this._isValid) return false;

  // No deterministic solutions, find cell with the fewest choices and try each one in turn
  // until success.
  var locChoice = this.findCellWithFewestChoices();
  if (locChoice.isEmpty()) return false;

  var cell = this.getCell(locChoice);
  var allowedValues = cell._allowed.allowedValuesArray();
  for (var i = 0; i < allowedValues.length; i++) {
    var val = allowedValues[i];
    var board = this.clone();
    if (board.trySolve(locChoice, val)) {
      board.copyTo(this);
      return true;
    }
  }

  return false;
};

// Count number of solutions up to a maximum cap (maxSolutions). Returns the number found (<= maxSolutions)
Board.prototype.countSolutions = function (maxSolutions) {
  if (!maxSolutions || maxSolutions < 1) maxSolutions = 2;
  var count = 0;

  function search(board) {
    // quick exit
    if (count >= maxSolutions) return;

    try {
      board.updateAllowed();
    } catch (e) {
      return;
    }
    if (!board._isValid) return;

    // apply deterministic moves
    while (board.acceptPossibles()) {
      try {
        board.updateAllowed();
      } catch (e) {
        return;
      }
      if (!board._isValid) return;
    }

    if (board._isSolved) {
      count++;
      return;
    }

    var locChoice = board.findCellWithFewestChoices();
    if (locChoice.isEmpty()) return;
    var cell = board.getCell(locChoice);
    var allowedValues = cell._allowed.allowedValuesArray();
    for (var i = 0; i < allowedValues.length; i++) {
      if (count >= maxSolutions) return;
      var val = allowedValues[i];
      var clone = board.clone();
      try {
        var ccell = clone.getCell(locChoice);
        if (!ccell.isAllowed(val)) continue;
        ccell.setValue(val);
      } catch (e) {
        // skip invalid assignment
        continue;
      }
      search(clone);
    }
  }

  try {
    var startBoard = this.clone();
    search(startBoard);
  } catch (e) {
    // ignore errors and return whatever count we got
  }
  return count;
};

Board.prototype.toString = function () {
  var text = "";
  for (var row = 0; row < BoardSize; row++)
    for (var col = 0; col < BoardSize; col++) {
      var val = this._digits[row][col].getValue();
      text += val == 0 || val == null ? "." : String(val);
    }
  return text;
};

Board.prototype.setString = function (value) {
  // Assumes all input is digits 1..9 or ./space
  if (value.length != BoardSize * BoardSize) return false; //Input string is not of length 81
  var n = 0;
  for (var row = 0; row < BoardSize; row++)
    for (var col = 0; col < BoardSize; col++) {
      var ch = parseInt(value.charAt(n++)); // converts '0' to 0 etc
      var cell = this._digits[row][col];
      cell.setGiven(!isNaN(ch) ? ch : 0);
    }
  this.updateAllowed();
  return true;
};

// Serialize the board state including per-cell notes into a plain object
Board.prototype.serialize = function () {
  var obj = {};
  obj.serial = this.toString();
  obj.notes = [];
  for (var r = 0; r < BoardSize; r++) {
    for (var c = 0; c < BoardSize; c++) {
      try {
        var cell = this.getCell(new CellLocation(r, c));
        obj.notes.push(
          cell && cell._notes && typeof cell._notes._mask !== "undefined"
            ? cell._notes._mask
            : 0
        );
      } catch (e) {
        obj.notes.push(0);
      }
    }
  }
  return obj;
};

// Deserialize a plain object previously produced by serialize().
// Returns { ok: boolean, foundNotes: boolean }
Board.prototype.deserialize = function (obj) {
  if (!obj || !obj.serial || obj.serial.length !== BoardSize * BoardSize)
    return { ok: false, foundNotes: false };
  var ok = this.setString(obj.serial);
  var foundNotes = false;
  if (
    ok &&
    obj.notes &&
    Array.isArray(obj.notes) &&
    obj.notes.length === BoardSize * BoardSize
  ) {
    var idx = 0;
    for (var r = 0; r < BoardSize; r++) {
      for (var c = 0; c < BoardSize; c++) {
        var mask = Number(obj.notes[idx++]) || 0;
        try {
          var cell = this.getCell(new CellLocation(r, c));
          if (cell) {
            cell._notes = new AllowedValues(mask);
            if (mask !== 0) foundNotes = true;
          }
        } catch (e) {
          /* ignore per-cell errors */
        }
      }
    }
  }
  // ensure allowed values are updated after deserializing
  try {
    this.updateAllowed();
  } catch (e) {}
  return { ok: !!ok, foundNotes: !!foundNotes };
};

// Fisher-Yates shuffle helper for arrays
function _shuffleArray(arr) {
  for (var i = arr.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

// Generate a full valid Sudoku solution by randomized backtracking.
// After this returns true, the Board will be completely filled (and valid).
Board.prototype.generateFullSolution = function () {
  // operate in-place
  this.clear();

  var self = this;

  function fill(board) {
    board.updateAllowed();
    if (!board._isValid) return false;
    if (board._isSolved) return true;

    // Choose a cell to try: collect all unassigned cells with the minimum
    // number of candidates and pick one at random. This avoids deterministic
    // behavior when many cells have the same candidate count (e.g. all 9 at start).
    var gridLocs = CellLocation.grid();
    var minCount = 10;
    var minLocs = [];
    for (var li = 0; li < gridLocs.length; li++) {
      var l = gridLocs[li];
      var cc = board.getCell(l);
      if (!cc.isAssigned()) {
        var cnt = cc.getAllowedClone().count();
        if (cnt < minCount) {
          minCount = cnt;
          minLocs = [l];
        } else if (cnt === minCount) {
          minLocs.push(l);
        }
      }
    }
    if (minLocs.length === 0) return false;
    // If any cell has 0 candidates, this branch is invalid
    if (minCount === 0) return false;
    var loc = minLocs[Math.floor(Math.random() * minLocs.length)];
    var cell = board.getCell(loc);
    var opts = cell._allowed.allowedValuesArray();
    _shuffleArray(opts);
    for (var i = 0; i < opts.length; i++) {
      var v = opts[i];
      var clone = board.clone();
      try {
        clone.getCell(loc).setValue(v);
      } catch (e) {
        continue;
      }
      if (fill(clone)) {
        clone.copyTo(board);
        return true;
      }
    }
    return false;
  }

  return fill(this);
};

// Generate a puzzle by creating a full solution, then removing digits while preserving uniqueness.
// minClues: desired minimum number of givens to keep (between 17 and 81). If omitted, defaults to 30.
// Returns an object { ok: boolean, clues: number }
Board.prototype.generatePuzzle = function (minClues) {
  if (!minClues || typeof minClues !== "number") minClues = 30;
  if (minClues < 17) minClues = 17; // practical minimum for uniqueness
  if (minClues > 81) minClues = 81;

  // Step 1: produce a complete valid solution
  var ok = this.generateFullSolution();
  if (!ok) return { ok: false, clues: 0 };

  // Ensure all cells are marked as givens initially
  for (var r = 0; r < BoardSize; r++)
    for (var c = 0; c < BoardSize; c++) {
      try {
        var cell = this.getCell(new CellLocation(r, c));
        var val = cell.getValue();
        cell.setGiven(val);
      } catch (e) {}
    }

  // Build a shuffled list of all positions to attempt removal
  var positions = [];
  for (var i = 0; i < BoardSize * BoardSize; i++) positions.push(i);
  _shuffleArray(positions);

  // Try to remove digits one by one while keeping uniqueness (stop when we have minClues left)
  var total = BoardSize * BoardSize;
  for (var idx = 0; idx < positions.length && total > minClues; idx++) {
    var p = positions[idx];
    var row = Math.floor(p / BoardSize);
    var col = p % BoardSize;
    var loc = new CellLocation(row, col);
    var saved = this.getCell(loc).getValue();
    if (saved === 0) continue;

    // Temporarily remove the value
    var backup = this.clone();
    try {
      this.getCell(loc).setValue(0);
      // After clearing, run uniqueness check on a clone to avoid modifying current board
      var test = this.clone();
      var count = 0;
      try {
        count = test.countSolutions(2);
      } catch (e) {
        count = 2; // assume multiple if counting failed
      }
      if (count === 1) {
        // removal valid; decrease total
        total--;
      } else {
        // revert
        backup.copyTo(this);
      }
    } catch (e) {
      // revert on any error
      try {
        backup.copyTo(this);
      } catch (e2) {}
    }
  }

  // Mark current non-zero cells as givens
  var clues = 0;
  for (var r2 = 0; r2 < BoardSize; r2++)
    for (var c2 = 0; c2 < BoardSize; c2++) {
      var cell2 = this.getCell(new CellLocation(r2, c2));
      if (cell2.getValue() !== 0) {
        cell2.setGiven(cell2.getValue());
        clues++;
      }
    }

  // Final updateAllowed
  try {
    this.updateAllowed();
  } catch (e) {}

  return { ok: true, clues: clues };
};
